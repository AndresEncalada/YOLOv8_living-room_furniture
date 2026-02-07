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
def generate_mixed_yaml(feedback_multiplier=1, data_limit=None):
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
        'names': {0: 'Sofa', 1: 'Rug', 2: 'Pillowss'}
    }
    
    os.makedirs(TEMP_YOLO_DIR, exist_ok=True)
    with open(yaml_path, 'w') as f:
        yaml.dump(data_config, f)
        
    return yaml_path

# --- 4. CICLO DE ENTRENAMIENTO ---
# --- 4. CICLO DE ENTRENAMIENTO (CON CARGA DESDE MLFLOW) ---
def execute_retraining_cycle(base_model_path=None, data_limit=None):
   
    try:
        if mlflow.active_run(): mlflow.end_run()

        # --- 1. CONFIGURACIÓN DE MLFLOW ---
        db_path = os.path.join(BASE_DIR, "mlflow.db")
        tracking_uri = f"sqlite:///{db_path.replace(os.sep, '/')}"
        mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.tracking.MlflowClient()

        logger.info(f"🚀 Iniciando Retraining (Límite: {data_limit})...")

        # --- 2. BUSCAR ÚLTIMA VERSIÓN EN MLFLOW ---
        model_to_train_path = base_model_path 
        
        try:
            logger.info(f"🔍 Buscando última versión de '{REGISTERED_MODEL_NAME}' en MLflow...")
            # Buscamos todas las versiones registradas
            versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
            
            if versions:
                versions.sort(key=lambda x: int(x.version), reverse=True)
                latest_version = versions[0]
                run_id = latest_version.run_id
                
                logger.info(f"📥 Última versión encontrada: v{latest_version.version} (Run ID: {run_id})")
                
                # Descargamos los pesos de esa versión a una carpeta temporal
                download_dir = os.path.join(TEMP_YOLO_DIR, "downloads")
                os.makedirs(download_dir, exist_ok=True)
                
                # Listamos artefactos para encontrar el .pt dentro de la carpeta 'weights'
                artifacts = client.list_artifacts(run_id, "weights")
                pt_files = [x.path for x in artifacts if x.path.endswith(".pt")]
                
                if pt_files:
                    artifact_path = pt_files[0]
                    logger.info(f"   Descargando artefacto: {artifact_path}...")
                    
                    local_path = mlflow.artifacts.download_artifacts(
                        run_id=run_id, 
                        artifact_path=artifact_path, 
                        dst_path=download_dir
                    )
                    model_to_train_path = local_path 
                    logger.info(f"✅ Modelo cargado desde MLflow: {model_to_train_path}")
                else:
                    logger.warning("⚠️ No se encontró archivo .pt en el Run de MLflow. Usando ruta local.")
            else:
                logger.warning("⚠️ No hay versiones registradas en MLflow. Usando ruta local.")
                
        except Exception as e:
            logger.error(f"❌ Error al intentar cargar desde MLflow: {e}")

        # Validación final del modelo
        if not model_to_train_path or not os.path.exists(model_to_train_path):
            logger.error(f"❌ No se encontró un modelo válido para entrenar (ni en MLflow ni local).")
            return None

        # --- 3. PREPARAR DATOS Y MODELO ---
        # Aumentamos el feedback_multiplier a 10 para forzar aprendizaje
        train_yaml = generate_mixed_yaml(feedback_multiplier=10, data_limit=data_limit)
        
        model = YOLO(model_to_train_path)
        
        # --- 4. CONFIGURACIÓN DEL EXPERIMENTO ---
        experiment_name = "Furniture_Continuous_Learning"
        try:
            mlflow.create_experiment(
                name=experiment_name,
                artifact_location=f"file:///{MLFLOW_ARTIFACTS_PATH.replace(os.sep, '/')}"
            )
        except:
            pass
        mlflow.set_experiment(experiment_name)
        
        # Preparamos nombre para la NUEVA versión
        dst_weights_path, new_filename = get_next_version_path()
        run_name_dynamic = f"Retrain_{new_filename.replace('.pt', '')}"

        # --- 5. EJECUCIÓN DEL ENTRENAMIENTO ---
        with mlflow.start_run(run_name=run_name_dynamic) as run:
            logger.info(f"🏋️‍♂️ Entrenando... Destino final será: {new_filename}")
            
            model.train(
                data=train_yaml,
                epochs=5, 
                imgsz=640,
                batch=8,            
                project=TEMP_YOLO_DIR,
                name='retrain_run',
                exist_ok=True,
                warmup_epochs=0,    
                freeze=10,          
                lr0=0.001,         
                lrf=0.01,           
                optimizer='AdamW', 
                
                dropout=0.0,
                mosaic=1.0,         
                
                plots=False,
                verbose=True,
                val=True,
                workers=0
            )
            
            # --- 6. REGISTRO DE RESULTADOS ---
            metrics = model.metrics
            map50 = metrics.box.map50
            map50_95 = metrics.box.map
            logger.info(f"📈 Nuevo mAP50: {map50:.4f}")
            logger.info(f"📈 Nuevo mAP50-95: {map50_95:.4f}")
            
            # Copiar pesos a la carpeta local 'models/'
            src_weights = os.path.join(TEMP_YOLO_DIR, "retrain_run", "weights", "best.pt")
            shutil.copy(src_weights, dst_weights_path)
            
            # Registrar en MLflow
            mlflow.log_metric("map50", map50)
            mlflow.log_metric("map50-95", map50_95)
            mlflow.log_artifact(dst_weights_path, artifact_path="weights")
            
            # Registrar nueva versión del modelo
            try: client.create_registered_model(REGISTERED_MODEL_NAME)
            except: pass
            
            mv = client.create_model_version(
                name=REGISTERED_MODEL_NAME,
                source=f"runs:/{run.info.run_id}/weights/{new_filename}",
                run_id=run.info.run_id
            )

            logger.info(f"✨ Versión registrada oficialmente en MLflow: v{mv.version}")
            
            return run.info.run_id

    except Exception as e:
        logger.error(f"❌ Error crítico en retraining: {e}")
        import traceback
        traceback.print_exc()
        return None