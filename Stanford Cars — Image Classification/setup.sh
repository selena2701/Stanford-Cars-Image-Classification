#!/bin/bash
# Setup script for Streamlit deployment
# This script prepares the environment for deployment

echo "🚀 Setting up Stanford Cars Classification App..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "📌 Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Check if artifacts directory exists
if [ ! -d "artifacts" ]; then
    echo "⚠️  Warning: artifacts/ directory not found!"
    echo "   Make sure to run the notebook to generate required artifacts."
fi

# Check if model file exists
if [ ! -f "artifacts/best_model.pth" ]; then
    echo "⚠️  Warning: artifacts/best_model.pth not found!"
    echo "   The app requires a trained model to function."
fi

# Check if class mappings exist
if [ ! -f "class_mappings.json" ]; then
    echo "⚠️  Warning: class_mappings.json not found!"
    echo "   This file is required for the app to work."
fi

echo "✅ Setup complete!"
echo ""
echo "To run the app locally:"
echo "  streamlit run streamlit_app.py"
echo ""
echo "To deploy to Streamlit Cloud:"
echo "  1. Push this repository to GitHub"
echo "  2. Go to https://share.streamlit.io"
echo "  3. Connect your repository"
echo "  4. Set main file path: Stanford Cars — Image Classification/streamlit_app.py"

