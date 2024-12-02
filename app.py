from fastapi import FastAPI, File, UploadFile, Request #type:ignore
from fastapi.responses import HTMLResponse #type:ignore
from fastapi.templating import Jinja2Templates #type:ignore
from fastapi.staticfiles import StaticFiles #type:ignore
from torch_snippets import *
from skimage.transform import resize 
from ultralytics import YOLO
import numpy as np 
import cv2
import datetime
import random
import os

app = FastAPI()
model = YOLO("yolo11n.pt")
model2 = YOLO("yolo11n-seg.pt")


app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request, "Message": "Love me like you do yeah~"})

@app.post("/upload")
async def upload(request: Request, file: UploadFile = File(...)):
    os.makedirs("static/uploaded", exist_ok=True)
    os.makedirs("static/saved", exist_ok=True)


    img_path = os.path.join("static/uploaded", file.filename)
    with open(img_path, "wb") as f:
        f.write(await file.read())

    img = cv2.imread(img_path) 
    results = model(img)        
    boxes = results[0].boxes.xyxy.cpu().numpy()  
    scores = results[0].boxes.conf.cpu().numpy() 

    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{score:.2f}"
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    processed_img_path = os.path.join("static/saved", file.filename)
    cv2.imwrite(processed_img_path, img)

    return templates.TemplateResponse(
        "home.html",
        {
            "request": request,
            "Message": "Image Uploaded and Processed!",
            "uploaded_img_path": f"/static/uploaded/{file.filename}",
            "processed_img_path": f"/static/saved/{file.filename}",
        }
    )


@app.post("/real_time")
async def upload(request: Request):
    cam = cv2.VideoCapture(2)
    cv2.namedWindow("test")
    img_counter = 0     

    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break

        res = model(frame)[0].plot()
        cv2.imshow("test", res)

        k = cv2.waitKey(1)
        if k%256 == 27:
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            img_name = "img_{}.png".format(img_counter)
            img_name = f"static/saved/{img_name}"
            cv2.imwrite(img_name, frame)
            show(read(img_name), title="{} written!".format(img_name))
            img_counter += 1

    cam.release()
    cv2.destroyAllWindows()

    return templates.TemplateResponse(
        "home.html",
        {
            "request": request,
            "Ms": "Executed Successfully!"
        }
    )
    
@app.post("/real_time2")
async def upload(request: Request):
    cam = cv2.VideoCapture(2) 

    alpha = 0.5 

    class_colors = {
        "person": [0, 0, 255], 
        'cell phone': [0, 255, 0], 
        "chair": [255, 0, 0],  
    }

    def get_class_color(class_name):
        return class_colors.get(class_name, [random.randint(0, 255) for _ in range(3)])

    while True:
        ret, frame = cam.read()  
        if not ret:
            print("Cam not found")
            break

        frame_resized = cv2.resize(frame, (1024, 768))  

        res = model2(frame_resized)[0]  

        if res.masks is not None and len(res.masks) > 0:
            highlighted_frame = frame.copy() 

            for i in range(len(res.masks)):
                mask = res.masks.data[i].cpu().numpy() 
                mask_binary = (mask > 0).astype(int)  

                mask_resized = resize(mask_binary, (frame.shape[0], frame.shape[1]), mode='reflect', anti_aliasing=True)

                class_idx = int(res.boxes[i].cls)
                class_name = res.names[class_idx]  


                color = get_class_color(class_name)  
                mask_colored = np.zeros_like(frame)  
                mask_colored[mask_resized > 0] = color  

                highlighted_frame[mask_resized > 0] = cv2.addWeighted(frame, 1 - alpha, mask_colored, alpha, 0)[mask_resized > 0]

            cv2.imshow('Camera', highlighted_frame)
        else:
            cv2.imshow('Camera', frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break

        if key == ord(" "): 
            filename = f'{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.png'
            cv2.imwrite(f'static\saved\{filename}', frame)
            cv2.imshow('Saved', frame)
            #show(read(f'static\saved\{filename}'))
            print(f"Image saved as {filename}")

    cam.release()
    cv2.destroyAllWindows()

    return templates.TemplateResponse(  
            "home.html",
            {
                "request": request,
                "Ms2": "Executed Successfully!"
            }
        )