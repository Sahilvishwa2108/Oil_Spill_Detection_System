# Lightweight Dockerfile for HuggingFace Spaces
FROM python:3.10-slim

WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip cache purge

# Copy application files (models excluded via .dockerignore)
COPY main.py .
COPY download_models.py .
COPY README.md .

# Create necessary directories
RUN mkdir -p models uploads

# Download models at runtime (smaller build)
# Models will be downloaded when the app starts

# Set environment variables for HuggingFace Spaces
ENV PYTHONUNBUFFERED=1
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

# Expose HuggingFace Spaces port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Run with model download and reduced memory usage
CMD ["sh", "-c", "python download_models.py && uvicorn main:app --host 0.0.0.0 --port 7860 --workers 1"]
