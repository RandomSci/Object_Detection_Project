from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request, "Message": "Love me like you do yeah~"})

@app.post("/upload")
async def upload(request: Request, file: UploadFile = File(...)):
    os.makedirs("static/uploaded", exist_ok=True)

    img_path = os.path.join("static/uploaded", file.filename)

    with open(img_path, "wb") as f:
        f.write(await file.read())

    result = "Pogi"
    
    return templates.TemplateResponse(
        "home.html", 
        {"request": request, 
        "Message": "Image Uploaded!", 
        "img_path": f"/static/uploaded/{file.filename}", 
        "result": f"Projected Object: {result}"}
    )
