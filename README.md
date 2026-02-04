# Furniture Detection System with Continuous Learning

This project implements an end-to-end **Computer Vision** solution for detecting furniture (Sofas, Rugs, Cushions) in living rooms. Unlike a static model, this system integrates an **MLOps** cycle and **Active Learning**, allowing the model to improve automatically through user feedback.

## Technologies and Libraries Used

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-blue?style=for-the-badge)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Jinja2](https://img.shields.io/badge/Jinja2-B41717?style=for-the-badge&logo=jinja&logoColor=white)

* **Ultralytics YOLOv8:** Base model for object detection.
* **FastAPI & Uvicorn:** Asynchronous server to expose the model via REST API.
* **MLflow:** Model lifecycle management, experiment tracking, and versioning.
* **Roboflow:** Dataset management and versioning.
* **OpenCV & Matplotlib:** Image processing and visualization.

---

## Method Overview

The system is not limited to performing inferences, but implements a **Continuous Improvement** workflow:

1.  **Cold Start:** It starts with a base model (`yolov8n.pt`) trained on a small subset of the complete Roboflow dataset (*Living Room Object Detection*).
2.  **Inference (API):** The user uploads an image through the web interface. The model detects objects and returns coordinates (Bounding Boxes).
3.  **Feedback Loop:** If detection is incorrect, the system allows capturing the image and corrected label, storing it in a *Feedback Dataset*.
4.  **Re-training (Background Task):** A background service combines the processed dataset with new feedback data, performs *Fine-Tuning* of the model, and registers the new version in MLflow.
5.  **Hot-Swap:** The API automatically updates the model in production without stopping the service.

---

## Project Structure

```text
yolov8_living-room_furniture/
│
├── app/                        # Main application source code
│   ├── templates/              # HTML templates for Frontend
│   │   └── index.html          # User interface (Upload/Predict)
│   ├── main.py                 # FastAPI server (API Endpoints)
│   └── retrain_service.py      # MLOps logic and automatic Re-training
│
├── data/                       # Data storage (Ignored in git)
│   ├── base_dataset/           # Base dataset downloaded from Roboflow
│   ├── feedback_dataset/       # Images collected from user
│   ├── processed_dataset/      # Processed base dataset
│   └── yolo_retrain_work/      # Temporary training files
│
├── models/                     # Store of trained weights
│   ├── best.pt                 # Initial base model
│   └── best_vX.pt              # Versions generated after re-training
│
├── Notebooks/                  # Experimentation and Analysis
│   ├── 01_EDA_Dataset_Base.ipynb      # Exploratory Data Analysis
│   ├── 02_Entrenamiento_Inicial.ipynb # Base model training (Sabotage/Full)
│   └── 03_Prediction.ipynb            # Inference testing and Retrain simulation
│
├── mlruns/                     # MLflow artifacts and metrics (Local)
├── requirements.txt            # Project dependencies
├── run_app.bat                 # Script to start FastAPI server
├── setup_gpu.bat               # Script to configure CUDA/Torch environment
└── .gitignore                  
```

## Installation and Execution

### 1. Clone the repository and install dependencies
```bash
git clone <URL_DEL_REPO>
cd yolov8_living-room_furniture

python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```
### 2. Run the Application

You can use the automatic script on Windows:
```bash
run_app.bat
```
Or execute manually with Uvicorn:

```bash
uvicorn app.main:app --reload
```

### 3. Access the Web
Open the browser at: `localhost:8000`

---

## Authors

* **Karen Ortiz** - [Github](https://github.com/Karenop4)  
  Contact: +593 99 444 1682 - karenorpe2004@gmail.com
* **Andrés Encalada** - [Github](https://github.com/AndresEncalada)  
  Contact: +593 98 358 6619 - andres23102004@gmail.com
