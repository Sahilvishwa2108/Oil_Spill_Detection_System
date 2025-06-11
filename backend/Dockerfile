# Optimized Backend Dockerfile for Memory Efficiency
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (minimal)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements and install Python dependencies with optimizations
COPY requirements.txt .
RUN pip install --no-cache-dir --no-deps -r requirements.txt \
    && pip cache purge

# Copy application code
COPY . .

# Create models directory
RUN mkdir -p models

# Download models on startup (Railway will handle this)
# Models will be downloaded automatically from Hugging Face

# Expose port (Railway will set PORT environment variable)
EXPOSE 8000

# Run the application with Railway's dynamic port
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
