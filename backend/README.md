# Oil Spill Detection API - Backend

## Deployment on Render (Free Tier)

This backend is optimized for deployment on Render's free tier with GitHub Student Developer Pack benefits.

### Features
- **Memory Optimized**: Models download on startup, not included in build
- **Free Tier Compatible**: Uses efficient resource management
- **Auto-scaling**: Lazy loading of ML models
- **Cross-Origin Support**: Configured for Vercel frontend

### Quick Deploy to Render

1. **Fork/Push to GitHub**: Ensure your code is in a GitHub repository

2. **Connect to Render**:
   - Go to [render.com](https://render.com)
   - Sign up with GitHub Student Pack
   - Create new Web Service
   - Connect your repository

3. **Configuration**:
   - **Build Command**: `echo "Building..."`
   - **Start Command**: `python main.py`
   - **Environment**: Docker
   - **Dockerfile Path**: `./backend/Dockerfile`
   - **Auto-Deploy**: Disabled (manual deploy recommended)

4. **Environment Variables**:
   ```
   HUGGINGFACE_REPO=sahilvishwa2108/oil-spill-detection-models
   CORS_ORIGINS=https://your-frontend.vercel.app
   PORT=10000
   ```

5. **Health Check**: `/health`

### Model Management
- Models (~220MB total) download automatically on first startup
- Cached for subsequent requests
- Lazy loading reduces memory usage

### API Endpoints
- `GET /health` - Health check
- `GET /` - API info
- `GET /models/info` - Model status
- `POST /predict` - Image prediction

### Troubleshooting
- **Build Timeout**: Models download at runtime, not build time
- **Memory Issues**: Only one model loads at a time
- **CORS Errors**: Update CORS_ORIGINS environment variable

### Local Development
```bash
cd backend
pip install -r requirements.txt
python main.py
```

Visit `http://localhost:8000/docs` for API documentation.
