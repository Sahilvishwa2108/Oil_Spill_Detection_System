# 🚀 Production Deployment Guide

This guide will walk you through deploying your Oil Spill Detection System to production using your GitHub Student Developer Pack.

## 📋 Prerequisites

Before starting, make sure you have:
- [ ] GitHub account (with Student Developer Pack activated)
- [ ] Azure account (free with Student Pack)
- [ ] Vercel account (free tier)
- [ ] Your models uploaded to Hugging Face Hub

## 🎯 Deployment Architecture

```
Frontend (Vercel) → Backend (Azure App Service) → Models (Hugging Face Hub)
```

## Step 1: Prepare Your Hugging Face Repository

1. **Upload your models to Hugging Face:**
   ```bash
   # Install Hugging Face CLI
   pip install huggingface_hub
   
   # Login to Hugging Face
   huggingface-cli login
   
   # Create repository
   huggingface-cli repo create sahilvishwa2108/oil-spill-detection-models --type model
   
   # Upload models
   huggingface-cli upload sahilvishwa2108/oil-spill-detection-models backend/models/deeplab_final_model.h5 deeplab_final_model.h5
   huggingface-cli upload sahilvishwa2108/oil-spill-detection-models backend/models/unet_final_model.h5 unet_final_model.h5
   ```

## Step 2: Deploy Backend to Azure

1. **Create Azure App Service:**
   - Go to [Azure Portal](https://portal.azure.com)
   - Create new "App Service"
   - Choose Python 3.10 runtime
   - Name: `oil-spill-backend`

2. **Configure deployment:**
   - Go to Deployment Center
   - Connect to your GitHub repository
   - Select the main branch
   - Choose GitHub Actions for CI/CD

3. **Set Environment Variables in Azure:**
   ```
   ENVIRONMENT=production
   HUGGINGFACE_REPO=sahilvishwa2108/oil-spill-detection-models
   CORS_ORIGINS=https://your-frontend.vercel.app
   WEBSITE_PORT=8000
   ```

## Step 3: Deploy Frontend to Vercel

1. **Connect to Vercel:**
   - Go to [Vercel](https://vercel.com)
   - Import your GitHub repository
   - Select the `frontend` folder as root directory

2. **Configure Environment Variables:**
   ```
   NEXT_PUBLIC_API_URL=https://oil-spill-backend.azurewebsites.net
   NEXT_PUBLIC_ENVIRONMENT=production
   ```

3. **Update CORS in Backend:**
   - Update your Azure App Service environment variable:
   ```
   CORS_ORIGINS=https://your-actual-vercel-url.vercel.app
   ```

## Step 4: Set up GitHub Actions Secrets

Add these secrets to your GitHub repository (Settings → Secrets and variables → Actions):

### Azure Secrets:
```
AZURE_CREDENTIALS: {
  "clientId": "your-client-id",
  "clientSecret": "your-client-secret",
  "subscriptionId": "your-subscription-id",
  "tenantId": "your-tenant-id"
}
```

### Vercel Secrets:
```
VERCEL_TOKEN: your-vercel-token
VERCEL_ORG_ID: your-org-id
VERCEL_PROJECT_ID: your-project-id
```

## Step 5: Test Your Deployment

1. **Backend Health Check:**
   ```
   curl https://oil-spill-backend.azurewebsites.net/health
   ```

2. **Frontend Access:**
   ```
   https://your-project.vercel.app
   ```

## 🎉 Success!

Your Oil Spill Detection System is now live! 

**Live URLs for your resume:**
- 🌐 **Frontend:** https://your-project.vercel.app
- 🔧 **Backend API:** https://oil-spill-backend.azurewebsites.net
- 📚 **API Docs:** https://oil-spill-backend.azurewebsites.net/docs

## 🔧 Troubleshooting

### Common Issues:

1. **Models not loading:**
   - Check Hugging Face repository permissions
   - Verify environment variables

2. **CORS errors:**
   - Update CORS_ORIGINS in Azure App Service
   - Restart the app service

3. **Slow startup:**
   - Models download on first startup (this is normal)
   - Subsequent startups will be faster

## 📊 Monitoring

- **Azure:** Use Application Insights for backend monitoring
- **Vercel:** Built-in analytics and performance monitoring
- **GitHub Actions:** Check deployment status in Actions tab

## 💰 Cost Estimation

With GitHub Student Pack:
- **Azure App Service:** $0 (Free tier for students)
- **Vercel:** $0 (Free tier)
- **Hugging Face:** $0 (Public repository)
- **GitHub Actions:** $0 (Free tier minutes)

**Total Cost: $0/month** 🎉
