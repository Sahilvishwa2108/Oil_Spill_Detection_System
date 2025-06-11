#!/bin/bash
# Deploy to Hugging Face Spaces Script

echo "🚀 Setting up Hugging Face Spaces deployment..."

# Check if huggingface_hub is available
if ! command -v huggingface-cli &> /dev/null; then
    echo "Installing Hugging Face CLI..."
    pip install huggingface_hub
fi

echo "📝 Instructions for Manual Deployment:"
echo "========================================="
echo ""
echo "1. Create a new Space on Hugging Face:"
echo "   - Go to: https://huggingface.co/new-space"
echo "   - Name: oil-spill-detection-api"
echo "   - SDK: Docker"
echo "   - Hardware: CPU Basic (Free)"
echo ""
echo "2. Clone your new space:"
echo "   git clone https://huggingface.co/spaces/YOUR_USERNAME/oil-spill-detection-api"
echo ""
echo "3. Copy backend files to the space:"
echo "   - Copy all files from backend/ folder"
echo "   - Make sure README.md has the --- header"
echo ""
echo "4. Push to deploy:"
echo "   git add ."
echo "   git commit -m 'Deploy Oil Spill Detection API'"
echo "   git push"
echo ""
echo "✨ Your API will be available at:"
echo "   https://YOUR_USERNAME-oil-spill-detection-api.hf.space"
