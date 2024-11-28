from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os
from torch_snippets import *
from ultralytics import YOLO
import cv2

app = FastAPI()
model = YOLO("yolo11n.pt")

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
    cam = cv2.VideoCapture(0)
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