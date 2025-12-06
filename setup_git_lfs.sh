#!/bin/bash
# Quick setup script for Git LFS to handle large model files

echo "🔧 Setting up Git LFS for large model files..."

# Check if Git LFS is installed
if ! command -v git-lfs &> /dev/null; then
    echo "❌ Git LFS is not installed!"
    echo ""
    echo "Please install Git LFS first:"
    echo "  macOS:   brew install git-lfs"
    echo "  Windows: Download from https://git-lfs.github.com/"
    echo "  Linux:   sudo apt-get install git-lfs"
    echo ""
    exit 1
fi

echo "✅ Git LFS is installed: $(git lfs version)"

# Initialize Git LFS
echo ""
echo "📦 Initializing Git LFS..."
git lfs install

# Track .pth files (PyTorch model files)
echo "🎯 Tracking .pth model files..."
git lfs track "**/*.pth"
git lfs track "Stanford Cars — Image Classification/artifacts/*.pth"

# Add .gitattributes
echo "📝 Adding .gitattributes..."
git add .gitattributes

# Check if model file exists
MODEL_PATH="Stanford Cars — Image Classification/artifacts/best_model.pth"
if [ -f "$MODEL_PATH" ]; then
    echo ""
    echo "✅ Found model file: $MODEL_PATH"
    echo ""
    read -p "Do you want to add and commit the model file now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "📤 Adding model file to Git LFS..."
        git add "$MODEL_PATH"
        echo "✅ Model file added to staging area"
        echo ""
        echo "Next steps:"
        echo "  1. git commit -m 'Add model file via Git LFS'"
        echo "  2. git push origin main"
    else
        echo "⏭️  Skipping model file. You can add it later with:"
        echo "  git add '$MODEL_PATH'"
    fi
else
    echo "⚠️  Model file not found: $MODEL_PATH"
    echo "   Make sure the model file exists before committing."
fi

echo ""
echo "✅ Git LFS setup complete!"
echo ""
echo "To verify, run: git lfs ls-files"

