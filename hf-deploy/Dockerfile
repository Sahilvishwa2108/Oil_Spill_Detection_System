# Optimized Dockerfile for Hugging Face Spaces
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
RUN pip install --no-cache-dir -r requirements.txt \
    && pip cache purge

# Copy application code
COPY . .

# Create models directory
RUN mkdir -p models

# Expose Hugging Face Spaces port
EXPOSE 7860

# Run the application with Hugging Face Spaces port
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
