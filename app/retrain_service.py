import os
import shutil
import yaml
import logging
import torch
import time
from ultralytics import YOLO

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RetrainService")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
BASE_DATASET = os.path.join(DATA_DIR, "base_dataset")
FEEDBACK_DATASET = os.path.join(DATA_DIR, "feedback_dataset")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(os.path.join(FEEDBACK_DATASET, "images"), exist_ok=True)
os.makedirs(os.path.join(FEEDBACK_DATASET, "labels"), exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def find_dataset_folders(search_root):
    train_dir = None
    val_dir = None
    for root, dirs, files in os.walk(search_root):
        if 'images' in dirs:
            full_path = os.path.join(root, 'images').replace('\\', '/')
            parent_name = os.path.basename(root).lower()
            if parent_name in ['train', 'training']:
                train_dir = full_path
            elif parent_name in ['valid', 'val', 'validation', 'test']:
                if not val_dir or parent_name in ['valid', 'val']:
                    val_dir = full_path
    return train_dir, val_dir

def generate_combined_yaml():
    original_yaml_path = None
    for root, dirs, files in os.walk(BASE_DATASET):
        if "data.yaml" in files:
            original_yaml_path = os.path.join(root, "data.yaml")
            break
            
    if not original_yaml_path: raise FileNotFoundError("data.yaml not found")

    with open(original_yaml_path, 'r') as f:
        base_config = yaml.safe_load(f)

    abs_base_train, abs_base_val = find_dataset_folders(BASE_DATASET)
    abs_feedback_train = os.path.abspath(os.path.join(FEEDBACK_DATASET, "images")).replace('\\', '/')
    
    feedback_count = len(os.listdir(abs_feedback_train))
    
    if feedback_count > 0:
        logger.info(f"⚡ FORCE MODE: Entrenando SOLO con {feedback_count} imágenes de feedback.")
        train_paths = [abs_feedback_train]
        val_path = abs_feedback_train 
    else:
        logger.warning("No feedback found. Using base dataset.")
        train_paths = [abs_base_train]
        val_path = abs_base_val if abs_base_val else abs_base_train

    retrain_config = {
        'path': '', 
        'train': train_paths,
        'val': val_path,
        'names': base_config.get('names'),
        'nc': base_config.get('nc')
    }

    output_yaml = os.path.join(DATA_DIR, "retrain_config.yaml")
    with open(output_yaml, 'w') as f:
        yaml.dump(retrain_config, f)
        
    return output_yaml

def execute_retraining_cycle(current_model_path):
    logger.info("Starting STABLE retraining cycle...")
    
    USE_DEVICE = 0 if torch.cuda.is_available() else 'cpu'
    
    try:
        yaml_path = generate_combined_yaml()
        
        if not os.path.exists(current_model_path):
            raise FileNotFoundError(f"Current model missing: {current_model_path}")
            
        model = YOLO(current_model_path)
        model.train(
            data=yaml_path,
            epochs=25,       
            imgsz=640,
            batch=8,
            device=USE_DEVICE,
            name='retrain_run',
            project=os.path.join(BASE_DIR, "models_history"),
            exist_ok=True,
            
            optimizer='SGD', 
            lr0=0.01,      
            lrf=0.01,     
            momentum=0.9,
            weight_decay=0.0005,

            warmup_epochs=0,
            freeze=0,
            mosaic=0.0,
            verbose=True,
            plots=False,
            amp=False
        )
        
        trained_weights = os.path.join(BASE_DIR, "models_history", "retrain_run", "weights", "best.pt")
        
        if os.path.exists(trained_weights):
            timestamp = int(time.time())
            new_filename = f"best_v{timestamp}.pt"
            new_model_path = os.path.join(MODEL_DIR, new_filename)
            shutil.copy(trained_weights, new_model_path)
            logger.info(f"New model saved at: {new_model_path}")
            return new_model_path
        else:
            return None

    except Exception as e:
        logger.error(f"Error: {e}")
        return None

def clear_feedback_data():
    try:
        shutil.rmtree(os.path.join(FEEDBACK_DATASET, "images"))
        shutil.rmtree(os.path.join(FEEDBACK_DATASET, "labels"))
        os.makedirs(os.path.join(FEEDBACK_DATASET, "images"), exist_ok=True)
        os.makedirs(os.path.join(FEEDBACK_DATASET, "labels"), exist_ok=True)
        return True
    except Exception as e:
        return False