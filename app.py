# app.py
import os
import io
import base64
from typing import List, Dict, Any

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np

app = FastAPI(title="YOLO Inference API")

MODEL_PATH = os.environ.get("MODEL_PATH", "best.pt")
DEVICE = os.environ.get("DEVICE", "cpu")  # "cpu" or "0" for GPU 0

# Load model (if not found we keep server running and show error on /)
try:
    model = YOLO(MODEL_PATH)
    print(f"Loaded model from {MODEL_PATH}")
except Exception as e:
    print(f"Failed to load model from {MODEL_PATH}: {e}")
    model = None

def draw_boxes_pil(image: Image.Image, boxes: List[Dict[str, Any]]) -> Image.Image:
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default()
    except:
        font = None
    for b in boxes:
        x1, y1, x2, y2 = b["bbox"]
        label = f"{b['class_name']}:{b['confidence']:.2f}"
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        if font:
            draw.text((x1, max(0, y1 - 10)), label, fill="red", font=font)
        else:
            draw.text((x1, max(0, y1 - 10)), label, fill="red")
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded on server.")
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    np_img = np.array(img)

    # Run inference (device can be 'cpu' or '0' etc.)
    results = model.predict(source=np_img, imgsz=640, conf=0.25, device=DEVICE)
    r = results[0]

    boxes = []
    for box in r.boxes:
        xyxy = box.xyxy.cpu().numpy().tolist()[0]
        conf = float(box.conf.cpu().numpy()[0]) if hasattr(box.conf, "cpu") else float(box.conf)
        cls = int(box.cls.cpu().numpy()[0]) if hasattr(box.cls, "cpu") else int(box.cls)
        name = model.names[int(cls)] if model and hasattr(model, "names") else str(cls)
        boxes.append({
            "bbox": [float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])],
            "confidence": conf,
            "class_id": cls,
            "class_name": name
        })

    out_img = draw_boxes_pil(img.copy(), boxes)
    buffered = io.BytesIO()
    out_img.save(buffered, format="JPEG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode()

    return JSONResponse({
        "predictions": boxes,
        "image_with_boxes_base64": img_b64,
    })

@app.get("/")
def root():
    return {"status": "ok", "model_loaded": model is not None, "model_path": MODEL_PATH}
