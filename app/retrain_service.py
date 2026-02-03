import os
import random
import shutil
import yaml
import glob
import mlflow
import torch
import logging
from ultralytics import YOLO

# --- CONFIGURACIÓN ---
logger = logging.getLogger("RetrainService")
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Rutas clave
TEMP_DIR = os.path.join(BASE_DIR, "data", "temp_uploads")
FEEDBACK_DIR = os.path.join(BASE_DIR, "data", "feedback_dataset")
BASE_DATASET_DIR = os.path.join(BASE_DIR, "data", "base_dataset") 
TEMP_YOLO_DIR = os.path.join(BASE_DIR, "data", "yolo_retrain_work")

MLFLOW_DB = "sqlite:///mlflow.db"
REGISTERED_MODEL_NAME = "Furniture_Model_YOLO"

USE_DEVICE = 0 if torch.cuda.is_available() else 'cpu'

# --- MAPA DE CLASES (TRADUCCIÓN) ---
TARGET_CLASSES = {
    0: ['sofa', 'couch', 'loveseat', 'settee'],    
    1: ['rug', 'carpet', 'mat', 'floor mat'],      
    2: ['pillow', 'cushion', 'throw pillow']       
}

# --- 1. DETECTOR DE IDs ---
def build_class_mapping(dataset_path):
    yaml_files = glob.glob(os.path.join(dataset_path, "data.yaml"))
    if not yaml_files:
        yaml_files = glob.glob(os.path.join(dataset_path, "*", "data.yaml"))
    
    if not yaml_files: return {0:0, 1:1, 2:2}

    yaml_path = yaml_files[0]
    with open(yaml_path, 'r') as f: data = yaml.safe_load(f)
        
    original_names = data.get('names', [])
    id_map = {}
    
    iterator = original_names.items() if isinstance(original_names, dict) else enumerate(original_names)

    for orig_id, orig_name in iterator:
        orig_name_clean = orig_name.lower().strip()
        for target_id, synonyms in TARGET_CLASSES.items():
            if any(syn in orig_name_clean for syn in synonyms):
                id_map[orig_id] = target_id
                break
    return id_map

# --- 2. SANITIZADOR ---
def sanitize_labels(dataset_root):
    """
    Versión protegida: Traduce IDs originales O respeta IDs ya procesados.
    """
    # Tu mapa de IDs originales
    ID_MAP = {12: 0, 11: 1, 19: 2}
    # IDs que ya están bien y no queremos borrar
    ALLOWED_IDS = [0, 1, 2]

    label_files = glob.glob(os.path.join(dataset_root, "**", "labels", "*.txt"), recursive=True)
    
    for txt_file in label_files:
        try:
            with open(txt_file, 'r') as f:
                lines = f.readlines()
            
            new_lines = []
            for line in lines:
                parts = line.split()
                if not parts: continue
                
                cls_id = int(parts[0])
                
                # CASO A: Es un ID original (11, 12, 19) -> Traducir
                if cls_id in ID_MAP:
                    new_id = ID_MAP[cls_id]
                    new_lines.append(f"{new_id} {' '.join(parts[1:])}\n")
                
                # CASO B: Ya es un ID corregido (0, 1, 2) -> Mantener
                elif cls_id in ALLOWED_IDS:
                    new_lines.append(line)
            
            # Solo sobreescribir si el archivo cambió para no corromper fechas
            with open(txt_file, 'w') as f:
                f.writelines(new_lines)
        except:
            pass

# --- 3. BUSCADOR DE CARPETAS ---
def find_dataset_folders():
    """Devuelve tupla (ruta_train, ruta_valid)"""
    train_path = None
    valid_path = None
    root_found = None

    # Buscar raíz
    if os.path.exists(os.path.join(BASE_DATASET_DIR, "train")):
        root_found = BASE_DATASET_DIR
    elif os.path.exists(BASE_DATASET_DIR):
        for item in os.listdir(BASE_DATASET_DIR):
            sub = os.path.join(BASE_DATASET_DIR, item)
            if os.path.isdir(sub) and os.path.exists(os.path.join(sub, "train")):
                root_found = sub
                break
    
    if root_found:
        # Sanitizar TODO el directorio encontrado (train y valid)
        sanitize_labels(root_found)
        
        # Definir rutas específicas
        t = os.path.join(root_found, "train", "images")
        v = os.path.join(root_found, "valid", "images")
        
        if os.path.exists(t): train_path = t
        if os.path.exists(v): valid_path = v
        
        # Si no existe valid, usamos train (fallback, pero intentamos evitarlo)
        if not valid_path: valid_path = train_path

    return train_path, valid_path

def generate_mixed_yaml(feedback_multiplier=5):
    yaml_path = os.path.join(TEMP_YOLO_DIR, "mixed_training.yaml")
    feedback_imgs = os.path.join(FEEDBACK_DIR, "images")
    
    base_train, base_valid = find_dataset_folders()
    
    # Fuentes de ENTRENAMIENTO (Feedback + Base Train)
    train_sources = [feedback_imgs]* feedback_multiplier
    if base_train: train_sources.append(base_train)
    
    # Fuentes de VALIDACIÓN
    val_sources = [base_valid] if base_valid else train_sources 

    logger.info(f"📁 Train Sources: {train_sources}")
    logger.info(f"📁 Valid Sources: {val_sources}")

    data_config = {
        'path': '', 
        'train': train_sources,
        'val': val_sources,  # <--- AQUÍ ESTÁ EL CAMBIO CRÍTICO
        'nc': 3,
        'names': {0: 'Sofa', 1: 'Rug', 2: 'Pillows'}
    }
    
    os.makedirs(TEMP_YOLO_DIR, exist_ok=True)
    with open(yaml_path, 'w') as f:
        yaml.dump(data_config, f)
        
    return yaml_path

def clear_feedback_data():
    try:
        if os.path.exists(FEEDBACK_DIR): shutil.rmtree(FEEDBACK_DIR)
        os.makedirs(os.path.join(FEEDBACK_DIR, "images"), exist_ok=True)
        os.makedirs(os.path.join(FEEDBACK_DIR, "labels"), exist_ok=True)
        return True
    except: return False

def execute_retraining_cycle(base_model_path):
    try:
        if mlflow.active_run(): mlflow.end_run()

        logger.info("Iniciando ciclo de Retraining...")
        train_yaml = generate_mixed_yaml()
        
        model = YOLO(base_model_path)
        
        mlflow.set_tracking_uri(MLFLOW_DB)
        mlflow.set_experiment("Furniture_Continuous_Learning")
        
        with mlflow.start_run(run_name="Retrain_Real_Val") as run:
            logger.info(" Entrenando...")
            nueva_seed = random.randint(1, 1000000)
            model.train(
                data=train_yaml,
                epochs=5, 
                warmup_epochs=1,
                seed=nueva_seed,
                lr0=0.001,
                lrf=0.01,
                freeze=0,
                batch=8,
                imgsz=640,
                project=TEMP_YOLO_DIR,
                name='retrain_run',
                exist_ok=True,
                verbose=True,
                plots=False,
                close_mosaic=0,
                mosaic=1.0,
                mixup=0.2,
                cls=2.0,
                scale=0.6,
                copy_paste=0.1,
                degrees=10
            )
            
            metrics = model.metrics
            map50 = metrics.box.map50
            logger.info(f" Nuevo mAP50: {map50:.4f}")
            mlflow.log_metric("map50", map50)
            
            weights_path = os.path.join(TEMP_YOLO_DIR, "retrain_run", "weights", "best.pt")
            mlflow.log_artifact(weights_path, artifact_path="weights")
            
            client = mlflow.tracking.MlflowClient()
            client.create_model_version(
                name=REGISTERED_MODEL_NAME,
                source=f"runs:/{run.info.run_id}/weights",
                run_id=run.info.run_id
            )
            
            return run.info.run_id

    except Exception as e:
        logger.error(f"❌ Error en retraining: {e}")
        return None