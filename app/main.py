import os
import cv2
import shutil
import uuid
import numpy as np
import logging
import gc
import glob
from typing import List, Dict, Any
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Body
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from ultralytics import YOLO
import app.retrain_service as retrain_service

# --- LOGGING ---
class EndpointFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "/api/status" not in record.getMessage()
logging.getLogger("uvicorn.access").addFilter(EndpointFilter())

# --- CONFIG ---
app = FastAPI(title="Furniture Detection System", version="3.5.0")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("API")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(BASE_DIR, "models")
TEMP_DIR = os.path.join(BASE_DIR, "data", "temp_uploads")
FEEDBACK_DIR = os.path.join(BASE_DIR, "data", "feedback_dataset")

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(os.path.join(FEEDBACK_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(FEEDBACK_DIR, "labels"), exist_ok=True)

TARGET_CLASSES = {
    "sofa", "sofas", "rug", "rugs", 
    "pillow", "pillows", "pillowss", "cushion", "cushions"
}

templates = Jinja2Templates(directory="app/templates")

# --- GLOBAL STATE ---
# Initial search for any .pt file in models/
initial_models = glob.glob(os.path.join(MODEL_DIR, "*.pt"))
# If exists, take the last one, else assume best.pt
CURRENT_MODEL_PATH = initial_models[-1] if initial_models else os.path.join(MODEL_DIR, "best.pt")

model = None
is_training = False

@app.on_event("startup")
def load_model():
    global model, CURRENT_MODEL_PATH
    if os.path.exists(CURRENT_MODEL_PATH):
        model = YOLO(CURRENT_MODEL_PATH)
        logger.info(f"Loaded: {os.path.basename(CURRENT_MODEL_PATH)}")
    else:
        logger.warning(f"Model not found at {CURRENT_MODEL_PATH}")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/predict")
async def predict_batch(files: List[UploadFile] = File(...)):
    global model
    if model is None: raise HTTPException(503, "Model not loaded")

    batch_results = []
    for file in files:
        file_ext = file.filename.split('.')[-1]
        file_id = f"{uuid.uuid4()}.{file_ext}"
        temp_path = os.path.join(TEMP_DIR, file_id)
        
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        img = cv2.imread(temp_path)
        if img is None: continue
            
        results = model(img)
        detections = []
        for r in results:
            for box in r.boxes:
                class_name = model.names[int(box.cls)]
                if class_name.lower() in TARGET_CLASSES:
                    detections.append({
                        "class": class_name,
                        "class_id": int(box.cls),
                        "confidence": float(box.conf),
                        "box": [float(x) for x in box.xyxy[0]]
                    })

        batch_results.append({
            "file_id": file_id,
            "original_name": file.filename,
            "detections": detections
        })
    return JSONResponse(content={"results": batch_results})

@app.post("/api/feedback")
async def save_feedback(payload: Dict[str, Any] = Body(...)):
    file_id = payload.get("file_id")
    corrected_boxes = payload.get("boxes", []) 
    src_path = os.path.join(TEMP_DIR, file_id)
    if not os.path.exists(src_path): raise HTTPException(404, "Image source not found")
        
    dst_img_path = os.path.join(FEEDBACK_DIR, "images", file_id)
    shutil.move(src_path, dst_img_path)
    
    img = cv2.imread(dst_img_path)
    h, w, _ = img.shape
    label_path = os.path.join(FEEDBACK_DIR, "labels", file_id.rsplit('.', 1)[0] + ".txt")
    
    with open(label_path, "w") as f:
        for item in corrected_boxes:
            x1, y1, x2, y2 = item['box']
            cls_id = item['class_id']
            w_box = x2 - x1
            h_box = y2 - y1
            cx = x1 + (w_box / 2)
            cy = y1 + (h_box / 2)
            f.write(f"{cls_id} {cx/w:.6f} {cy/h:.6f} {w_box/w:.6f} {h_box/h:.6f}\n")
            
    return {"status": "saved"}

@app.post("/api/reset")
def reset_dataset():
    if retrain_service.clear_feedback_data():
        return {"status": "cleared"}
    else:
        raise HTTPException(500, "Failed to clear")

# --- DYNAMIC RELOAD LOGIC ---
def background_retrain_task():
    global is_training, model, CURRENT_MODEL_PATH
    is_training = True
    
    logger.info(f"Retraining base on: {os.path.basename(CURRENT_MODEL_PATH)}")
    new_model_path = retrain_service.execute_retraining_cycle(CURRENT_MODEL_PATH)
    
    if new_model_path and os.path.exists(new_model_path):
        logger.info(f"Switching to NEW model: {os.path.basename(new_model_path)}")
        old_model_path = CURRENT_MODEL_PATH
        del model
        gc.collect()
        CURRENT_MODEL_PATH = new_model_path
        model = YOLO(CURRENT_MODEL_PATH)
        if old_model_path != new_model_path and os.path.exists(old_model_path):
            try:
                os.remove(old_model_path)
                logger.info("Old model file deleted.")
            except:
                pass
                
        # Warmup
        model(np.zeros((100, 100, 3), dtype=np.uint8), verbose=False)
    else:
        logger.error("Retraining returned no new path.")
        
    is_training = False

@app.post("/api/retrain")
async def trigger_retrain(background_tasks: BackgroundTasks):
    global is_training
    if is_training: return JSONResponse(409, {"status": "Busy"})
    background_tasks.add_task(background_retrain_task)
    return {"status": "Accepted"}

@app.get("/api/status")
def get_status():
    return {"training_active": is_training}