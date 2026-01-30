import os
import shutil
import yaml
import logging
import torch
import time
import stat, errno
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RetrainService")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
FEEDBACK_DATASET = os.path.join(DATA_DIR, "feedback_dataset")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(os.path.join(FEEDBACK_DATASET, "images"), exist_ok=True)
os.makedirs(os.path.join(FEEDBACK_DATASET, "labels"), exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# --- UTILS ---
def handle_remove_readonly(func, path, exc):
    excvalue = exc[1]
    if func in (os.rmdir, os.remove, os.unlink) and excvalue.errno == errno.EACCES:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    else: raise

def generate_combined_yaml():
    abs_feedback_train = os.path.abspath(os.path.join(FEEDBACK_DATASET, "images")).replace('\\', '/')
    
    if len(os.listdir(abs_feedback_train)) == 0:
        raise FileNotFoundError("Sube imágenes y corrígelas antes de entrenar.")

    # ESTRATEGIA: TRANSFER LEARNING
    # Forzamos 3 clases. Esto hará que YOLO adapte el modelo de GitHub
    # a tu nueva realidad de Sofa/Rug/Pillows.
    retrain_config = {
        'path': '', 
        'train': [abs_feedback_train],
        'val': abs_feedback_train, 
        'nc': 3,  # <--- FUERZA EL CAMBIO DE CABEZA DEL MODELO
        'names': { 0: 'Sofa', 1: 'Rug', 2: 'Pillows' }
    }

    output_yaml = os.path.join(DATA_DIR, "retrain_config.yaml")
    with open(output_yaml, 'w') as f:
        yaml.dump(retrain_config, f)
    return output_yaml

def execute_retraining_cycle(current_model_path):
    logger.info(f"🚀 Iniciando Transfer Learning sobre {os.path.basename(current_model_path)}...")
    USE_DEVICE = 0 if torch.cuda.is_available() else 'cpu'
    
    try:
        yaml_path = generate_combined_yaml()
        
        # Cargamos el modelo de GitHub (con sus 12 clases raras)
        model = YOLO(current_model_path)
        
        model.train(
            data=yaml_path,
            epochs=20, # Un poco más para asentar el cambio de cabeza
            imgsz=480,
            batch=8,
            device=USE_DEVICE,
            name='retrain_run',
            project=os.path.join(BASE_DIR, "models_history"),
            exist_ok=True,
            
            # Parametros para Transfer Learning efectivo
            optimizer='auto', 
            lr0=0.01, # Learning rate inicial estándar para reaprender clasificación
            verbose=True,
            plots=False
        )
        
        trained_weights = os.path.join(BASE_DIR, "models_history", "retrain_run", "weights", "best.pt")
        
        if os.path.exists(trained_weights):
            timestamp = int(time.time())
            new_filename = f"best_v{timestamp}.pt"
            new_model_path = os.path.join(MODEL_DIR, new_filename)
            shutil.copy(trained_weights, new_model_path)
            logger.info(f"✅ Modelo evolucionado guardado: {new_filename}")
            return new_model_path
        else:
            return None
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return None

def clear_feedback_data():
    try:
        if os.path.exists(os.path.join(FEEDBACK_DATASET, "images")):
            shutil.rmtree(os.path.join(FEEDBACK_DATASET, "images"), ignore_errors=False, onerror=handle_remove_readonly)
        if os.path.exists(os.path.join(FEEDBACK_DATASET, "labels")):
            shutil.rmtree(os.path.join(FEEDBACK_DATASET, "labels"), ignore_errors=False, onerror=handle_remove_readonly)
        os.makedirs(os.path.join(FEEDBACK_DATASET, "images"), exist_ok=True)
        os.makedirs(os.path.join(FEEDBACK_DATASET, "labels"), exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Error borrando: {e}")
        return False