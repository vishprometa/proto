#!/bin/bash

# Build Proto standalone binary
echo "🚀 Building Proto standalone binary..."

# Check if PyInstaller is installed
if ! command -v pyinstaller >/dev/null 2>&1; then
    echo "Installing PyInstaller..."
    pip install pyinstaller
fi

# Build the binary
pyinstaller --onefile \
            --name proto \
            --distpath . \
            --clean \
            main.py

echo "✅ Binary built successfully!"
echo "📦 Binary location: ./proto"
echo ""
echo "To test:"
echo "  ./proto --help"
echo ""
echo "To create a release:"
echo "  1. Rename to proto-$(uname -s | tr '[:upper:]' '[:lower:]')-$(uname -m)"
echo "  2. Upload to GitHub releases"
