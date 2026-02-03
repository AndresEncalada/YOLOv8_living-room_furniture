import os
import random
import shutil
import yaml
import mlflow
import logging
import glob
from ultralytics import YOLO

# --- CONFIGURACIÓN ---
logger = logging.getLogger("RetrainService")
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Rutas clave
FEEDBACK_DIR = os.path.join(BASE_DIR, "data", "feedback_dataset")
PROCESSED_DATASET_DIR = os.path.join(BASE_DIR, "data", "processed_dataset")
TEMP_YOLO_DIR = os.path.join(BASE_DIR, "data", "yolo_retrain_work")
MODELS_DIR = os.path.join(BASE_DIR, "models")
SUBSET_DIR = os.path.join(BASE_DIR, "data", "temp_subset_dataset")

# --- CONFIGURACIÓN MLFLOW (CORREGIDA) ---
# Forzamos que los artifacts se guarden en la raíz, no en Notebooks/
MLFLOW_ARTIFACTS_PATH = os.path.join(BASE_DIR, "mlruns")
REGISTERED_MODEL_NAME = "Furniture_Model_YOLO"

# --- 1. GESTOR DE VERSIONES ---
def get_next_version_path():
    if not os.path.exists(MODELS_DIR): os.makedirs(MODELS_DIR)
    counter = 2
    while True:
        filename = f"best_v{counter}.pt"
        full_path = os.path.join(MODELS_DIR, filename)
        if not os.path.exists(full_path): return full_path, filename
        counter += 1

# --- 2. PREPARAR SUBSET ---
def prepare_subset_dataset(limit=None):
    train_src = os.path.join(PROCESSED_DATASET_DIR, "train", "images")
    valid_src = os.path.join(PROCESSED_DATASET_DIR, "valid", "images")
    
    if not limit:
        return train_src, valid_src

    logger.info(f"✂️ Creando subset de entrenamiento limitado a {limit} imágenes...")
    
    if os.path.exists(SUBSET_DIR): shutil.rmtree(SUBSET_DIR)
    subset_train_imgs = os.path.join(SUBSET_DIR, "train", "images")
    subset_train_lbls = os.path.join(SUBSET_DIR, "train", "labels")
    os.makedirs(subset_train_imgs, exist_ok=True)
    os.makedirs(subset_train_lbls, exist_ok=True)

    all_images = glob.glob(os.path.join(train_src, "*.jpg"))
    if not all_images: return None, None
    
    selected_images = random.sample(all_images, min(limit, len(all_images)))
    
    for img_path in selected_images:
        basename = os.path.basename(img_path)
        lbl_path = os.path.join(PROCESSED_DATASET_DIR, "train", "labels", basename.replace('.jpg', '.txt'))
        
        shutil.copy(img_path, os.path.join(subset_train_imgs, basename))
        if os.path.exists(lbl_path):
            shutil.copy(lbl_path, os.path.join(subset_train_lbls, basename.replace('.jpg', '.txt')))
            
    return subset_train_imgs, valid_src

# --- 3. GENERADOR DE YAML ---
def generate_mixed_yaml(feedback_multiplier=2, data_limit=None):
    yaml_path = os.path.join(TEMP_YOLO_DIR, "mixed_training.yaml")
    feedback_imgs = os.path.join(FEEDBACK_DIR, "images")
    
    base_train, base_valid = prepare_subset_dataset(data_limit)
    
    if not base_train:
        raise FileNotFoundError("No se encontró el dataset base.")

    train_sources = [base_train]
    
    if os.path.exists(feedback_imgs) and len(os.listdir(feedback_imgs)) > 0:
        logger.info(f"♻️ Incorporando feedback (x{feedback_multiplier})...")
        train_sources.extend([feedback_imgs] * feedback_multiplier)
    
    data_config = {
        'path': '', 
        'train': train_sources,
        'val': base_valid,
        'nc': 3,
        'names': {0: 'Sofa', 1: 'Rug', 2: 'Pillows'}
    }
    
    os.makedirs(TEMP_YOLO_DIR, exist_ok=True)
    with open(yaml_path, 'w') as f:
        yaml.dump(data_config, f)
        
    return yaml_path

# --- 4. CICLO DE ENTRENAMIENTO ---
def execute_retraining_cycle(base_model_path, data_limit=None):
    try:
        if mlflow.active_run(): mlflow.end_run()

        logger.info(f"🚀 Iniciando Retraining (Límite: {data_limit})...")
        
        train_yaml = generate_mixed_yaml(data_limit=data_limit)
        
        if not os.path.exists(base_model_path):
            logger.error(f"Modelo base no encontrado: {base_model_path}")
            return None
            
        model = YOLO(base_model_path)
        
        # --- CONFIGURACIÓN MLFLOW ROBUSTA ---
        db_path = os.path.join(BASE_DIR, "mlflow.db")
        mlflow.set_tracking_uri(f"sqlite:///{db_path.replace(os.sep, '/')}")
        
        experiment_name = "Furniture_Continuous_Learning"
        
        # INTENTO DE CREACIÓN EXPLÍCITA CON RUTA CORRECTA
        try:
            mlflow.create_experiment(
                name=experiment_name,
                artifact_location=f"file:///{MLFLOW_ARTIFACTS_PATH.replace(os.sep, '/')}"
            )
            logger.info(f"✅ Experimento creado apuntando a: {MLFLOW_ARTIFACTS_PATH}")
        except:
            # Si ya existe, verificamos si está bien configurado (opcional) o seguimos
            pass

        mlflow.set_experiment(experiment_name)
        
        dst_weights_path, new_filename = get_next_version_path()
        run_name_dynamic = f"Retrain_{new_filename.replace('.pt', '')}"

        with mlflow.start_run(run_name=run_name_dynamic) as run:
            logger.info(f"🏋️‍♂️ Entrenando... Destino: {new_filename}")
            
            model.train(
                data=train_yaml,
                epochs=2,
                imgsz=640,
                batch=16,
                project=TEMP_YOLO_DIR,
                name='retrain_run',
                exist_ok=True,
                lr0=0.0001,
                lrf=0.1,
                dropout=0.0,
                mosaic=0.5,
                plots=False,
                verbose=True,
                val=True,
                warmup_epochs=0,
                workers=0
            )
            
            metrics = model.metrics
            map50 = metrics.box.map50
            logger.info(f"📈 Nuevo mAP50: {map50:.4f}")
            
            src_weights = os.path.join(TEMP_YOLO_DIR, "retrain_run", "weights", "best.pt")
            shutil.copy(src_weights, dst_weights_path)
            
            mlflow.log_metric("map50", map50)
            mlflow.log_artifact(dst_weights_path, artifact_path="weights")
            
            client = mlflow.tracking.MlflowClient()
            try: client.create_registered_model(REGISTERED_MODEL_NAME)
            except: pass
            
            mv = client.create_model_version(
                name=REGISTERED_MODEL_NAME,
                source=f"runs:/{run.info.run_id}/weights/{new_filename}",
                run_id=run.info.run_id
            )
            
            return run.info.run_id

    except Exception as e:
        logger.error(f"❌ Error en retraining: {e}")
        import traceback
        traceback.print_exc()
        return None