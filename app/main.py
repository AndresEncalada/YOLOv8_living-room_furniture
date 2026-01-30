import os
import cv2
import shutil
import uuid
import numpy as np
import logging
import gc
import glob
import torch
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

app = FastAPI(title="Furniture Detection System", version="5.0.0-GITHUB-SAVER")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("API")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(BASE_DIR, "models")
TEMP_DIR = os.path.join(BASE_DIR, "data", "temp_uploads")
FEEDBACK_DIR = os.path.join(BASE_DIR, "data", "feedback_dataset")

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(os.path.join(FEEDBACK_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(FEEDBACK_DIR, "labels"), exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

templates = Jinja2Templates(directory="app/templates")

# --- TRADUCTOR: MODELO GITHUB -> TU NEGOCIO ---
# Aquí mapeamos las clases raras del modelo de GitHub a tus 3 Clases.
CLASS_MAPPING = {
    "sofa": "Sofa",
    "couch": "Sofa",
    "accent chair": "Sofa",

    # RUG
    "rug": "Rug",
    "rugs": "Rug",
    "carpet": "Rug",
    "mat": "Rug",
    
    # PILLOWS
    "pillow": "Pillows",
    "pillows": "Pillows",
    "pillowss": "Pillows",
    "cushion": "Pillows",
    "cushions": "Pillows"
}

# --- CLASES EXTRA ---
EXTRA_ALLOWED_LABELS = {
    "chandelier": "Chandelier", 
    "chandeliers": "Chandelier",
    "lamp": "Lamp",
    "floor lamp": "Lamp",
    "table lamp": "Lamp",
    "plant": "Plant",
    "plants": "Plant",
    "plantss": "Plant", 
    "artificial plants": "Plant",
    "picture frame": "Decor",
    "coffee table": "Table",
    "end - side tables": "Table",
    "end table": "Table",
    "armchair": "Chair",
    "accent chair": "Chair"
}

MAIN_LABELS = {"Sofa", "Rug", "Pillows"}

model = None
is_training = False
CURRENT_MODEL_PATH = None

def cleanup_old_models(keep_file_path=None):
    try:
        all_models = glob.glob(os.path.join(MODEL_DIR, "best_v*.pt"))
        for m_path in all_models:
            if keep_file_path and os.path.abspath(m_path) == os.path.abspath(keep_file_path): continue
            try: os.remove(m_path)
            except: pass
    except: pass

@app.on_event("startup")
def load_model():
    global model, CURRENT_MODEL_PATH
    
    # 1. Buscamos modelos
    custom_models = glob.glob(os.path.join(MODEL_DIR, "*.pt"))
    
    # Ordenamos por fecha para usar siempre el último
    if custom_models:
        # Preferimos los que empiezan por 'best_v'
        latest_model = max(custom_models, key=os.path.getctime)
        CURRENT_MODEL_PATH = latest_model
        logger.info(f" Cargando MODELO PRE-ENTRENADO: {os.path.basename(latest_model)}")
        model = YOLO(latest_model)
    else:
        logger.warning(" No se encontró ningún modelo en /models. Descargando base...")
        CURRENT_MODEL_PATH = "yolov8m.pt"
        model = YOLO(CURRENT_MODEL_PATH)

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
            
        # Inferencia
        results = model(img, conf=0.10, iou=0.5)
        
        detections = []
        for r in results:
            for box in r.boxes:
                raw_name = model.names[int(box.cls)].lower()
                conf = float(box.conf)
                
                final_name = None
                is_main = False 

                # 1. Mapeo Inteligente
                if raw_name in CLASS_MAPPING:
                    final_name = CLASS_MAPPING[raw_name]
                    is_main = True
                
                # 2. Búsqueda Parcial (si dice "blue accent chair" -> "Sofa")
                elif any(k in raw_name for k in CLASS_MAPPING):
                     for k, v in CLASS_MAPPING.items():
                         if k in raw_name:
                             final_name = v
                             is_main = True
                             break

                # 3. Extras
                elif raw_name in EXTRA_ALLOWED_LABELS:
                    final_name = EXTRA_ALLOWED_LABELS[raw_name]
                    is_main = False
                elif any(k in raw_name for k in EXTRA_ALLOWED_LABELS):
                     for k, v in EXTRA_ALLOWED_LABELS.items():
                         if k in raw_name:
                             final_name = v
                             is_main = False
                             break
                
                # 4. Fallback (Si confiamos mucho)
                elif conf > 0.35:
                    final_name = raw_name.title()
                    is_main = False

                if final_name:
                    detections.append({
                        "class": final_name,
                        "class_id": int(box.cls), 
                        "confidence": conf,
                        "is_main": is_main, 
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
    
    # AL GUARDAR, UNIFICAMOS IDs
    ID_MAP = {"Sofa": 0, "Rug": 1, "Pillows": 2}

    with open(label_path, "w") as f:
        for item in corrected_boxes:
            cls_name = item['class']
            if cls_name in ID_MAP:
                cls_id = ID_MAP[cls_name]
                x1, y1, x2, y2 = item['box']
                w_box = x2 - x1
                h_box = y2 - y1
                cx = x1 + (w_box / 2)
                cy = y1 + (h_box / 2)
                f.write(f"{cls_id} {cx/w:.6f} {cy/h:.6f} {w_box/w:.6f} {h_box/h:.6f}\n")
            
    return {"status": "saved"}

@app.post("/api/reset")
def reset_dataset():
    if retrain_service.clear_feedback_data(): return {"status": "cleared"}
    else: raise HTTPException(500, "Failed to clear")

def background_retrain_task():
    global is_training, model, CURRENT_MODEL_PATH
    is_training = True
    new_model_path = retrain_service.execute_retraining_cycle(CURRENT_MODEL_PATH)
    if new_model_path and os.path.exists(new_model_path):
        del model
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        CURRENT_MODEL_PATH = new_model_path
        model = YOLO(CURRENT_MODEL_PATH)
        cleanup_old_models(keep_file_path=new_model_path)
        model(np.zeros((100, 100, 3), dtype=np.uint8), verbose=False)
    is_training = False

@app.post("/api/retrain")
async def trigger_retrain(background_tasks: BackgroundTasks):
    global is_training
    if is_training: return JSONResponse(409, {"status": "Busy"})
    background_tasks.add_task(background_retrain_task)
    return {"status": "Accepted"}

@app.get("/api/status")
def get_status(): return {"training_active": is_training}