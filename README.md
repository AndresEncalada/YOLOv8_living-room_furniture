# Sistema de Detección de Mobiliario con Aprendizaje Continuo

Este proyecto implementa una solución de **Visión Artificial** end-to-end para la detección de mobiliario (Sofás, Alfombras, Cojines) en salas de estar. A diferencia de un modelo estático, este sistema integra un ciclo de **MLOps** y **Aprendizaje Activo (Active Learning)**, permitiendo que el modelo mejore automáticamente mediante el feedback de los usuarios.

## Tecnologías y Librerías Utilizadas

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-blue?style=for-the-badge)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Jinja2](https://img.shields.io/badge/Jinja2-B41717?style=for-the-badge&logo=jinja&logoColor=white)

* **Ultralytics YOLOv8:** Modelo base para la detección de objetos.
* **FastAPI & Uvicorn:** Servidor asíncrono para exponer el modelo vía REST API.
* **MLflow:** Gestión del ciclo de vida del modelo, registro de experimentos y versionado.
* **Roboflow:** Gestión y versionado del dataset original.
* **OpenCV & Matplotlib:** Procesamiento de imágenes y visualización.

---

## Resumen del Método

El sistema no se limita a realizar inferencias, sino que implementa un flujo de **Mejora Continua**:

1.  **Cold Start:** Se inicia con un modelo base (`yolov8n.pt`) entrenado con un subset pequeño del dataset completo de Roboflow (*Living Room Object Detection*).
2.  **Inferencia (API):** El usuario sube una imagen a través de la interfaz web. El modelo detecta los objetos y devuelve las coordenadas (Bounding Boxes).
3.  **Feedback Loop:** Si la detección es incorrecta, el sistema permite capturar la imagen y la etiqueta corregida, almacenándola en un *Feedback Dataset*.
4.  **Re-entrenamiento (Background Task):** Un servicio en segundo plano combina el dataset procesado con los nuevos datos de feedback, ejecuta un *Fine-Tuning* del modelo y registra la nueva versión en MLflow.
5.  **Hot-Swap:** La API actualiza el modelo en producción automáticamente sin detener el servicio.

---

## Estructura del Proyecto

```text
yolov8_living-room_furniture/
│
├── app/                        # Código fuente de la aplicación principal
│   ├── templates/              # Plantillas HTML para el Frontend
│   │   └── index.html          # Interfaz de usuario (Upload/Predict)
│   ├── main.py                 # Servidor FastAPI (Endpoints API)
│   └── retrain_service.py      # Lógica de MLOps y Re-entrenamiento automático
│
├── data/                       # Almacenamiento de datos (Ignorado en git)
│   ├── base_dataset/           # Dataset base descargado de Robloflow
│   ├── feedback_dataset/       # Imágenes recolectadas del usuario
│   ├── processed_dataset/      # Dataset base procesado
│   └── yolo_retrain_work/      # Archivos temporales de entrenamiento
│
├── models/                     # Almacén de pesos entrenados
│   ├── best.pt                 # Modelo base inicial
│   └── best_vX.pt              # Versiones generadas tras el re-entrenamiento
│
├── Notebooks/                  # Experimentación y Análisis
│   ├── 01_EDA_Dataset_Base.ipynb      # Análisis Exploratorio de Datos
│   ├── 02_Entrenamiento_Inicial.ipynb # Entrenamiento del modelo base (Sabotaje/Full)
│   └── 03_Prediction.ipynb            # Pruebas de inferencia y simulación de Retrain
│
├── mlruns/                     # Artefactos y métricas de MLflow (Local)
├── requirements.txt            # Dependencias del proyecto
├── run_app.bat                 # Script para iniciar el servidor FastAPI
├── setup_gpu.bat               # Script para configurar entorno CUDA/Torch
└── .gitignore                  
```

## Instalación y Ejecución

### 1. Clonar el repositorio e instalar dependencias
```bash
git clone <URL_DEL_REPO>
cd yolov8_living-room_furniture

python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```
### 2. Ejecutar la Aplicación

Puede usar el script automático en Windows:
```bash
run_app.bat
```
O ejecutar manualmente con Uvicorn:

```bash
uvicorn app.main:app --reload
```

### 3. Acceder a la Web
Abrir el navegador en: `localhost:8000`

---

## Autores

* **Karen Ortiz** - [Github](https://github.com/Karenop4)  
  Contacto: +593 99 444 1682 - karenorpe2004@gmail.com
* **Andrés Encalada** - [Github](https://github.com/AndresEncalada)  
  Contacto: +593 98 358 6619 - andres23102004@gmail.com
