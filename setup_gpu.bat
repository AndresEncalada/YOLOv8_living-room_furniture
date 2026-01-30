@echo off
color 0A
echo ==============================================
echo   CONFIGURANDO ENTORNO GPU (NVIDIA RTX 3050)
echo ==============================================
echo.

echo [1/3] Deteniendo procesos de Python...
taskkill /IM python.exe /F 2>nul

echo.
echo [2/3] Desinstalando versiones actuales (CPU)...
pip uninstall torch torchvision torchaudio -y

echo.
echo [3/3] Descargando e Instalando PyTorch CUDA...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo.
echo ==============================================
echo   VERIFICACION FINAL
echo ==============================================
python -c "import torch; print('CUDA Disponible: ' + str(torch.cuda.is_available())); print('Dispositivo: ' + torch.cuda.get_device_name(0))"
echo.
pause