#!/bin/bash

# Build Proto installers for different platforms
# This script builds standalone executables using PyInstaller

set -e

echo "🚀 Building Proto standalone installers..."

# Create builds directory
mkdir -p builds

# Function to build for a specific platform
build_for_platform() {
    local platform=$1
    local arch=$2
    local output_name=$3
    
    echo "📦 Building for $platform ($arch)..."
    
    # Build with PyInstaller
    pyinstaller --onefile \
                --name proto \
                --distpath builds \
                --workpath build \
                --specpath build \
                --clean \
                main.py
    
    # Rename the output
    if [ -f "builds/proto" ]; then
        mv "builds/proto" "builds/$output_name"
        echo "✅ Built: builds/$output_name"
    else
        echo "❌ Build failed for $platform"
        return 1
    fi
}

# Build for current platform (macOS)
if [[ "$OSTYPE" == "darwin"* ]]; then
    if [[ $(uname -m) == "arm64" ]]; then
        build_for_platform "macOS" "ARM64" "proto-macos-arm64"
    else
        build_for_platform "macOS" "x64" "proto-macos-x64"
    fi
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    if [[ $(uname -m) == "aarch64" ]]; then
        build_for_platform "Linux" "ARM64" "proto-linux-arm64"
    else
        build_for_platform "Linux" "x64" "proto-linux-x64"
    fi
else
    echo "⚠️  Unsupported platform: $OSTYPE"
    echo "💡 To build for other platforms, run this script on those systems"
fi

# Create a simple installer script
cat > builds/install.sh << 'EOF'
#!/bin/bash

# Simple installer script for Proto
echo "🚀 Installing Proto..."

# Determine the correct binary for this system
if [[ "$OSTYPE" == "darwin"* ]]; then
    if [[ $(uname -m) == "arm64" ]]; then
        BINARY="proto-macos-arm64"
    else
        BINARY="proto-macos-x64"
    fi
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    if [[ $(uname -m) == "aarch64" ]]; then
        BINARY="proto-linux-arm64"
    else
        BINARY="proto-linux-x64"
    fi
else
    echo "❌ Unsupported platform: $OSTYPE"
    exit 1
fi

# Check if binary exists
if [ ! -f "$BINARY" ]; then
    echo "❌ Binary $BINARY not found!"
    echo "💡 Make sure you're running this script from the builds directory"
    exit 1
fi

# Make executable
chmod +x "$BINARY"

# Install to /usr/local/bin (requires sudo)
echo "📦 Installing to /usr/local/bin/proto..."
sudo cp "$BINARY" /usr/local/bin/proto

echo "✅ Proto installed successfully!"
echo "🎯 Run 'proto' to start using it!"
EOF

chmod +x builds/install.sh

echo ""
echo "🎉 Build complete! Available files:"
echo ""
ls -la builds/
echo ""
echo "📋 To install on this system:"
echo "   cd builds && ./install.sh"
echo ""
echo "📦 To distribute:"
echo "   - Upload builds/* to GitHub releases"
echo "   - Users can download and run directly"
echo ""
echo "💡 For other platforms, run this script on those systems"
