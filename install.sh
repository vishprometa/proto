#!/bin/bash

# Proto ClickHouse AI Agent Installer
# One-liner: curl -fsSL https://raw.githubusercontent.com/vishprometa/proto/main/install.sh | sh

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Detect OS and architecture
detect_system() {
    OS=$(uname -s)
    ARCH=$(uname -m)
    
    case $OS in
        "Darwin")
            if [[ "$ARCH" == "arm64" ]]; then
                BINARY="proto-darwin-arm64"
                PLATFORM="darwin-arm64"
            else
                BINARY="proto-darwin-x64"
                PLATFORM="darwin-x64"
            fi
            ;;
        "Linux")
            if [[ "$ARCH" == "aarch64" ]]; then
                BINARY="proto-linux-arm64"
                PLATFORM="linux-arm64"
            else
                BINARY="proto-linux-x64"
                PLATFORM="linux-x64"
            fi
            ;;
        *)
            print_error "Unsupported operating system: $OS"
            exit 1
            ;;
    esac
}

# Download and install Proto
install_proto() {
    print_status "Detecting system..."
    detect_system
    
    print_status "Installing Proto for $OS ($ARCH)..."
    
    # Create temporary directory
    TEMP_DIR=$(mktemp -d)
    cd "$TEMP_DIR"
    
    # Download the binary
    print_status "Downloading Proto binary..."
    # Download from GitHub releases
    DOWNLOAD_URL="https://github.com/vishprometa/proto/releases/latest/download/$BINARY"
    
    if ! curl -fsSL -o proto "$DOWNLOAD_URL"; then
        print_error "Failed to download Proto binary"
        print_warning "The binary release is not available yet."
        echo ""
        echo "For now, please install using pipx:"
        echo "  pipx install proto-clickhouse-agent"
        echo ""
        echo "Or install Python first, then run:"
        echo "  pip install proto-clickhouse-agent"
        exit 1
    fi
    
    # Make executable
    chmod +x proto
    
    # Install to /usr/local/bin
    print_status "Installing to /usr/local/bin/proto..."
    if ! sudo mv proto /usr/local/bin/; then
        print_error "Failed to install Proto. Make sure you have sudo privileges."
        exit 1
    fi
    
    # Clean up
    cd - > /dev/null
    rm -rf "$TEMP_DIR"
    
    print_success "Proto installed successfully!"
    print_status "Run 'proto' to start using it!"
    print_warning "First run will download the AI model (~3.5GB)"
}

# Check if already installed
check_existing() {
    if command -v proto >/dev/null 2>&1; then
        print_warning "Proto is already installed!"
        read -p "Do you want to reinstall? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_status "Installation cancelled."
            exit 0
        fi
        # Remove existing version
        sudo rm -f /usr/local/bin/proto 2>/dev/null || true
    fi
}

# Main installation flow
main() {
    echo "🚀 Proto ClickHouse AI Agent Installer"
    echo "======================================"
    echo
    
    check_existing
    install_proto
    
    echo
    print_success "Installation complete! 🎉"
    echo
    echo "Next steps:"
    echo "  1. Run 'proto' to start the interactive chat"
    echo "  2. Follow the onboarding to configure your setup"
    echo "  3. Visit https://github.com/vishprometa/proto for documentation"
}

# Run main function
main "$@"
