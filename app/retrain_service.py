import os
import random
import shutil
import yaml
import mlflow
import logging
import glob
from ultralytics import YOLO

# --- CONFIGURATION ---
logger = logging.getLogger("RetrainService")
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Key Paths
FEEDBACK_DIR = os.path.join(BASE_DIR, "data", "feedback_dataset")
PROCESSED_DATASET_DIR = os.path.join(BASE_DIR, "data", "processed_dataset")
TEMP_YOLO_DIR = os.path.join(BASE_DIR, "data", "yolo_retrain_work")
MODELS_DIR = os.path.join(BASE_DIR, "models")
SUBSET_DIR = os.path.join(BASE_DIR, "data", "temp_subset_dataset")

# --- MLFLOW CONFIGURATION ---
# Force artifacts to be saved in the root directory, not nested in Notebooks
MLFLOW_ARTIFACTS_PATH = os.path.join(BASE_DIR, "mlruns")
REGISTERED_MODEL_NAME = "Furniture_Model_YOLO"

# --- 1. VERSION MANAGER ---
def get_next_version_path():
    """
    Scans the local models directory to determine the next sequential filename
    (e.g., best_v2.pt, best_v3.pt) to avoid overwriting previous weights.
    """
    if not os.path.exists(MODELS_DIR): os.makedirs(MODELS_DIR)
    counter = 2
    while True:
        filename = f"best_v{counter}.pt"
        full_path = os.path.join(MODELS_DIR, filename)
        if not os.path.exists(full_path): return full_path, filename
        counter += 1

# --- 2. PREPARE SUBSET ---
def prepare_subset_dataset(limit=None):
    """
    Creates a temporary subset of the training data. 
    Useful for rapid prototyping or limited resource environments.
    
    Args:
        limit (int): Maximum number of images to sample from the source.
    """
    train_src = os.path.join(PROCESSED_DATASET_DIR, "train", "images")
    valid_src = os.path.join(PROCESSED_DATASET_DIR, "valid", "images")
    
    if not limit:
        return train_src, valid_src

    logger.info(f"✂️ Creating training subset limited to {limit} images...")
    
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

# --- 3. YAML GENERATOR ---
def generate_mixed_yaml(feedback_multiplier=1, data_limit=None):
    """
    Generates a dynamic YOLO dataset YAML file.
    
    Technical Note:
    It implements a 'Weighted Sampling' strategy by duplicating the feedback directory 
    in the source list multiple times (feedback_multiplier). This forces the model 
    to see corrected images more frequently during an epoch, prioritizing the 
    correction of previous errors.
    """
    yaml_path = os.path.join(TEMP_YOLO_DIR, "mixed_training.yaml")
    feedback_imgs = os.path.join(FEEDBACK_DIR, "images")
    
    base_train, base_valid = prepare_subset_dataset(data_limit)
    
    if not base_train:
        raise FileNotFoundError("Base dataset not found.")

    train_sources = [base_train]
    
    if os.path.exists(feedback_imgs) and len(os.listdir(feedback_imgs)) > 0:
        logger.info(f"♻️ Incorporating feedback (x{feedback_multiplier})...")
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

# --- 4. TRAINING CYCLE (WITH MLFLOW LOADING) ---
def execute_retraining_cycle(base_model_path=None, data_limit=None):
    """
    Orchestrates the full continuous learning pipeline:
    1. Downloads the latest production model from MLflow (if available).
    2. Prepares a new mixed dataset (Base + Feedback).
    3. Fine-tunes the model using transfer learning.
    4. Logs metrics and registers the new version back to MLflow.
    """
    try:
        
        
        if mlflow.active_run(): mlflow.end_run()

        # --- 1. MLFLOW SETUP ---
        db_path = os.path.join(BASE_DIR, "mlflow.db")
        tracking_uri = f"sqlite:///{db_path.replace(os.sep, '/')}"
        mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.tracking.MlflowClient()

        logger.info(f"🚀 Starting Retraining (Limit: {data_limit})...")

        # --- 2. FETCH LATEST VERSION FROM MLFLOW ---
        model_to_train_path = base_model_path 
        
        try:
            logger.info(f"🔍 Searching for latest version of '{REGISTERED_MODEL_NAME}' in MLflow...")
            # Search all registered versions
            versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
            
            if versions:
                versions.sort(key=lambda x: int(x.version), reverse=True)
                latest_version = versions[0]
                run_id = latest_version.run_id
                
                logger.info(f"📥 Latest version found: v{latest_version.version} (Run ID: {run_id})")
                
                # Download weights to a temporary directory
                download_dir = os.path.join(TEMP_YOLO_DIR, "downloads")
                os.makedirs(download_dir, exist_ok=True)
                
                # List artifacts to find the specific .pt file
                artifacts = client.list_artifacts(run_id, "weights")
                pt_files = [x.path for x in artifacts if x.path.endswith(".pt")]
                
                if pt_files:
                    artifact_path = pt_files[0]
                    logger.info(f"   Downloading artifact: {artifact_path}...")
                    
                    local_path = mlflow.artifacts.download_artifacts(
                        run_id=run_id, 
                        artifact_path=artifact_path, 
                        dst_path=download_dir
                    )
                    model_to_train_path = local_path 
                    logger.info(f"✅ Model loaded from MLflow: {model_to_train_path}")
                else:
                    logger.warning("⚠️ No .pt file found in MLflow Run. Using local path.")
            else:
                logger.warning("⚠️ No registered versions in MLflow. Using local path.")
                
        except Exception as e:
            logger.error(f"❌ Error attempting to load from MLflow: {e}")

        # Final model validation
        if not model_to_train_path or not os.path.exists(model_to_train_path):
            logger.error(f"❌ No valid model found for training (neither in MLflow nor local).")
            return None

        # --- 3. PREPARE DATA AND MODEL ---
        # We increase feedback_multiplier to 10 to force learning on corrections
        train_yaml = generate_mixed_yaml(feedback_multiplier=10, data_limit=data_limit)
        
        model = YOLO(model_to_train_path)
        
        # --- 4. EXPERIMENT SETUP ---
        experiment_name = "Furniture_Continuous_Learning"
        try:
            mlflow.create_experiment(
                name=experiment_name,
                artifact_location=f"file:///{MLFLOW_ARTIFACTS_PATH.replace(os.sep, '/')}"
            )
        except:
            pass
        mlflow.set_experiment(experiment_name)
        
        # Prepare name for the NEW version
        dst_weights_path, new_filename = get_next_version_path()
        run_name_dynamic = f"Retrain_{new_filename.replace('.pt', '')}"

        # --- 5. EXECUTE TRAINING ---
        with mlflow.start_run(run_name=run_name_dynamic) as run:
            logger.info(f"🏋️‍♂️ Training... Final destination will be: {new_filename}")
            
            model.train(
                data=train_yaml,
                epochs=5, 
                imgsz=640,
                batch=8,            
                project=TEMP_YOLO_DIR,
                name='retrain_run',
                exist_ok=True,
                warmup_epochs=0,    
                freeze=10,  # Technical: Freezes the first 10 layers (backbone) to preserve feature extraction logic.       
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
            
            # --- 6. LOGGING RESULTS ---
            metrics = model.metrics
            map50 = metrics.box.map50
            map50_95 = metrics.box.map
            logger.info(f"📈 New mAP50: {map50:.4f}")
            logger.info(f"📈 New mAP50-95: {map50_95:.4f}")
            
            # Copy weights to local 'models/' directory
            src_weights = os.path.join(TEMP_YOLO_DIR, "retrain_run", "weights", "best.pt")
            shutil.copy(src_weights, dst_weights_path)
            
            # Log to MLflow
            mlflow.log_metric("map50", map50)
            mlflow.log_metric("map50-95", map50_95)
            mlflow.log_artifact(dst_weights_path, artifact_path="weights")
            
            # Register new model version
            try: client.create_registered_model(REGISTERED_MODEL_NAME)
            except: pass
            
            mv = client.create_model_version(
                name=REGISTERED_MODEL_NAME,
                source=f"runs:/{run.info.run_id}/weights/{new_filename}",
                run_id=run.info.run_id
            )

            logger.info(f"✨ Version officially registered in MLflow: v{mv.version}")
            
            return run.info.run_id

    except Exception as e:
        logger.error(f"❌ Critical error in retraining: {e}")
        import traceback
        traceback.print_exc()
        return None
