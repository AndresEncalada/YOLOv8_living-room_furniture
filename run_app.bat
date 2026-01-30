@echo off
title YOLOv8 App
echo Instalando dependencias...
pip install -r requirements.txt
echo Iniciando Servidor...
uvicorn app.main:app --reload
pause