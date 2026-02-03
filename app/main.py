import sys
import os
import shutil
import uuid
import logging
import gc
import cv2
import torch
import mlflow
import threading  # <--- VITAL PARA LOS HILOS
from mlflow.tracking import MlflowClient
from typing import List, Dict, Any
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Body
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from ultralytics import YOLO
import app.retrain_service as retrain_service

# --- LOGGING & CONFIGURATION ---
class EndpointFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "/api/status" not in record.getMessage()

logging.getLogger("uvicorn.access").addFilter(EndpointFilter())
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("API")

app = FastAPI(title="Furniture Detection System", version="6.8.0-Threads-Reload")

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TEMP_DIR = os.path.join(BASE_DIR, "data", "temp_uploads")
FEEDBACK_DIR = os.path.join(BASE_DIR, "data", "feedback_dataset")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

# Ensure directories exist
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(os.path.join(FEEDBACK_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(FEEDBACK_DIR, "labels"), exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory="app/templates")

# MLflow Settings
MLFLOW_DB = "sqlite:///mlflow.db"
REGISTERED_MODEL_NAME = "Furniture_Model_YOLO"

# --- CLASS DEFINITIONS ---
STRICT_CLASSES = { 0: 'Sofa', 1: 'Rug', 2: 'Pillows' }
NAME_TO_ID = {v: k for k, v in STRICT_CLASSES.items()}
GENERIC_MAP = {
    'couch': 'Sofa', 'sofa': 'Sofa', 
    'rug': 'Rug', 'carpet': 'Rug', 'mat': 'Rug',
    'pillow': 'Pillows', 'cushion': 'Pillows'
}

# --- GLOBAL STATE & LOCKS ---
model = None
model_lock = threading.Lock() # <--- EL SEMÁFORO DE SEGURIDAD
is_training = False
CURRENT_VERSION_LABEL = "Unknown"

# --- HELPER: CORE PREDICTION LOGIC ---
def process_prediction(model_ref, img, file_id, original_name):
    """
    Lógica compartida para predecir (usada en upload y en reload).
    """
    is_generic_model = len(model_ref.names) > 10 
    
    results = model_ref(img, conf=0.25, iou=0.5)
    
    detections = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls)
            raw_conf = float(box.conf)
            raw_name = model_ref.names[cls_id].lower() if cls_id in model_ref.names else "unknown"
            final_name = None
            
            if is_generic_model:
                for key, val in GENERIC_MAP.items():
                    if key in raw_name:
                        final_name = val
                        break
            else:
                if cls_id in STRICT_CLASSES: final_name = STRICT_CLASSES[cls_id]
                else: final_name = raw_name.capitalize()

            if final_name:
                detections.append({
                    "class": final_name, "class_id": cls_id, 
                    "confidence": raw_conf, "is_main": True,
                    "box": [float(x) for x in box.xyxy[0]]
                })

    return {
        "file_id": file_id, 
        "original_name": original_name,
        "detections": detections
    }

# --- MODEL LOADING LOGIC ---
def load_model_from_mlflow(run_id, label):
    global model, CURRENT_VERSION_LABEL
    
    # PROTECCIÓN DE HILOS: Si ya se está cargando, rebotar la petición.
    if not model_lock.acquire(blocking=False):
        logger.warning("⚠️ Sistema ocupado cargando modelo. Intento ignorado.")
        return False

    try:
        mlflow.set_tracking_uri(MLFLOW_DB)
        client = MlflowClient()
        
        logger.info(f"📥 Descargando Run {run_id}...")
        local_path = client.download_artifacts(run_id, "weights", dst_path=TEMP_DIR)
        
        pt_file = os.path.join(local_path, "best.pt")
        if not os.path.exists(pt_file): pt_file = local_path 
        
        # Soft Delete: Limpiamos memoria sin destruir la variable abruptamente
        if model is not None:
            model = None 
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            
        model = YOLO(pt_file)
        CURRENT_VERSION_LABEL = label
        logger.info(f"✅ Modelo cambiado a: {label}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error cambiando modelo: {e}")
        return False
    finally:
        model_lock.release() # Liberar siempre el semáforo

@app.on_event("startup")
def load_initial_model():
    global model, CURRENT_VERSION_LABEL
    mlflow.set_tracking_uri(MLFLOW_DB)
    client = MlflowClient()
    
    logger.info("🚀 Iniciando sistema...")
    try:
        versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
        if versions:
            latest = sorted(versions, key=lambda x: int(x.version))[-1]
            local_path = client.download_artifacts(latest.run_id, "weights", dst_path=TEMP_DIR)
            pt_file = os.path.join(local_path, "best.pt")
            model = YOLO(pt_file)
            CURRENT_VERSION_LABEL = f"v{latest.version}"
            logger.info(f"⬇️ Cargado v{latest.version} desde MLflow.")
            return
    except Exception as e:
        logger.warning(f"⚠️ MLflow warning: {e}")

    local_model_path = os.path.join(BASE_DIR, "models", "best.pt")
    if os.path.exists(local_model_path):
        logger.info(f"📂 Cargando modelo local: {local_model_path}")
        model = YOLO(local_model_path)
        CURRENT_VERSION_LABEL = "Local Baseline"
    else:
        logger.error("❌ CRITICAL: No se encontró ningún modelo.")

# --- API ENDPOINTS ---

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/versions")
def get_model_versions():
    mlflow.set_tracking_uri(MLFLOW_DB)
    client = MlflowClient()
    try:
        versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
        results = []
        for v in versions:
            try:
                run = client.get_run(v.run_id)
                metrics = run.data.metrics
                
                # --- FIX ROBUSTO DEL mAP ---
                # Buscamos todas las variantes posibles para que no falle nunca
                map50 = metrics.get('map50') or \
                        metrics.get('metrics/mAP50(B)') or \
                        metrics.get('metrics/mAP50B') or \
                        metrics.get('val/map50') or \
                        0.0
                
                label = f"v{v.version} (mAP: {map50:.2f})"
                is_current = (f"v{v.version}" in CURRENT_VERSION_LABEL)
                
                results.append({
                    "run_id": v.run_id,
                    "version": v.version,
                    "label": label,
                    "current": is_current
                })
            except: continue 
        results.sort(key=lambda x: int(x['version']), reverse=True)
        return {"versions": results}
    except Exception:
        return {"versions": []}

@app.post("/api/switch-version")
def switch_version_endpoint(payload: Dict[str, str] = Body(...)):
    run_id = payload.get("run_id")
    label = payload.get("label", "Unknown")
    
    success = load_model_from_mlflow(run_id, label)
    if success:
        return {"status": "success", "new_version": label}
    elif not success and model is not None:
         raise HTTPException(409, "Server busy. Try again.")
    else:
         raise HTTPException(500, "Fatal error.")

@app.post("/api/predict")
async def predict_batch(files: List[UploadFile] = File(...)):
    global model
    if model is None: raise HTTPException(503, "Model not loaded")

    batch_results = []
    
    # Usamos el Lock para leer el modelo de forma segura
    with model_lock:
        local_model_ref = model
        
    if local_model_ref is None: raise HTTPException(503, "Model not ready")
    
    for file in files:
        file_ext = file.filename.split('.')[-1]
        file_id = f"{uuid.uuid4()}.{file_ext}"
        temp_path = os.path.join(TEMP_DIR, file_id)
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        img = cv2.imread(temp_path)
        if img is None: continue
            
        result = process_prediction(local_model_ref, img, file_id, file.filename)
        batch_results.append(result)

    return JSONResponse(content={"results": batch_results})

# --- NUEVO ENDPOINT PARA RECARGA AUTOMÁTICA ---
@app.post("/api/repredict")
def repredict_existing(payload: Dict[str, List[str]] = Body(...)):
    """
    Vuelve a predecir sobre imágenes ya subidas.
    """
    global model
    file_ids = payload.get("file_ids", [])
    if not file_ids: return {"results": []}

    with model_lock:
        local_model_ref = model
    
    if local_model_ref is None: raise HTTPException(503, "Model not loaded")

    batch_results = []
    for file_id in file_ids:
        temp_path = os.path.join(TEMP_DIR, file_id)
        if not os.path.exists(temp_path): continue
        img = cv2.imread(temp_path)
        if img is None: continue
        
        # Reutilizamos la lógica, pasando el file_id como nombre si no tenemos el original
        result = process_prediction(local_model_ref, img, file_id, file_id)
        batch_results.append(result)
        
    return JSONResponse(content={"results": batch_results})

@app.post("/api/feedback")
async def save_feedback(payload: Dict[str, Any] = Body(...)):
    file_id = payload.get("file_id")
    corrected_boxes = payload.get("boxes", []) 
    src_path = os.path.join(TEMP_DIR, file_id)
    if not os.path.exists(src_path): raise HTTPException(404, "Image missing")
    dst_img_path = os.path.join(FEEDBACK_DIR, "images", file_id)
    shutil.copy(src_path, dst_img_path) 
    img = cv2.imread(dst_img_path)
    h, w, _ = img.shape
    label_path = os.path.join(FEEDBACK_DIR, "labels", file_id.rsplit('.', 1)[0] + ".txt")
    with open(label_path, "w") as f:
        for item in corrected_boxes:
            cls_name = item['class']
            if cls_name in NAME_TO_ID:
                cls_id = NAME_TO_ID[cls_name]
                x1, y1, x2, y2 = item['box']
                w_box, h_box = x2 - x1, y2 - y1
                f.write(f"{cls_id} {(x1+w_box/2)/w:.6f} {(y1+h_box/2)/h:.6f} {w_box/w:.6f} {h_box/h:.6f}\n")
    return {"status": "saved"}

@app.post("/api/reset")
def reset_dataset():
    if retrain_service.clear_feedback_data(): return {"status": "cleared"}
    raise HTTPException(500, "Failed to clear")

def retrain_task_wrapper(temp_model_path):
    global is_training, model
    is_training = True
    try:
        new_run_id = retrain_service.execute_retraining_cycle(temp_model_path)
        if new_run_id:
            mlflow.set_tracking_uri(MLFLOW_DB)
            client = MlflowClient()
            versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
            if versions:
                latest = sorted(versions, key=lambda x: int(x.version))[-1]
                load_model_from_mlflow(new_run_id, f"v{latest.version}")
    except Exception as e:
        logger.error(f"Background training failed: {e}")
    finally:
        is_training = False
        if os.path.exists(temp_model_path): os.remove(temp_model_path)

@app.post("/api/retrain")
async def trigger_retrain(background_tasks: BackgroundTasks):
    global is_training, model
    if is_training: return JSONResponse(409, {"status": "Busy"})
    
    with model_lock: 
        temp_path = os.path.join(TEMP_DIR, "base_for_retrain.pt")
        model.save(temp_path)
    
    background_tasks.add_task(retrain_task_wrapper, temp_path)
    return {"status": "Accepted"}

@app.get("/api/status")
def get_status(): return {"training_active": is_training}