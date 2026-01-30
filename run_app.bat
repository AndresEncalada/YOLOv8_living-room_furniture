@echo off
title YOLOv8 App
echo Iniciando Servidor...
uvicorn app.main:app --reload
pause