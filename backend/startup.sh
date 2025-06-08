#!/bin/bash
# Startup script for Azure App Service

# Set environment variables for production
export ENVIRONMENT=production
export PORT=${PORT:-8000}

# Download models if they don't exist
if [ ! -f "models/unet_final_model.h5" ] || [ ! -f "models/deeplab_final_model.h5" ]; then
    echo "Downloading models from Hugging Face..."
    python download_models.py
    if [ $? -eq 0 ]; then
        echo "Models downloaded successfully"
    else
        echo "Warning: Model download failed, will use dummy models"
    fi
else
    echo "Models already exist, skipping download"
fi

# Start the application
echo "Starting Oil Spill Detection API on port $PORT..."
exec gunicorn --bind=0.0.0.0:$PORT --timeout 600 --workers=1 --worker-class=uvicorn.workers.UvicornWorker main:app
