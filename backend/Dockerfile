# Optimized Dockerfile for Render Deployment
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
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

# Copy application files
COPY main.py .

# Create models directory
RUN mkdir -p models

# Set environment variables for production
ENV PYTHONUNBUFFERED=1

# Start the application
CMD python main.py
