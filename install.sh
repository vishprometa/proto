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



# Download and install Proto
install_proto() {
    print_status "Checking system requirements..."
    
    # Check if Python is installed
    if ! command -v python3 >/dev/null 2>&1; then
        print_error "Python 3 is not installed!"
        echo ""
        echo "Please install Python 3 first:"
        echo ""
        echo "  macOS:"
        echo "    brew install python3"
        echo "    # or download from https://python.org"
        echo ""
        echo "  Linux (Ubuntu/Debian):"
        echo "    sudo apt update && sudo apt install python3 python3-pip"
        echo ""
        echo "  Linux (CentOS/RHEL):"
        echo "    sudo yum install python3 python3-pip"
        echo ""
        exit 1
    fi
    
    # Check Python version
    PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    
    if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
        print_error "Python 3.8 or higher is required. You have Python $PYTHON_VERSION"
        exit 1
    fi
    
    print_success "Python $PYTHON_VERSION detected"
    
    # Check if pipx is installed
    if ! command -v pipx >/dev/null 2>&1; then
        print_status "pipx not found. Installing pipx..."
        
        # Try to install pipx
        if python3 -m pip install --user pipx; then
            # Add pipx to PATH
            if python3 -m pipx ensurepath; then
                print_success "pipx installed successfully"
                # Reload shell environment
                export PATH="$HOME/.local/bin:$PATH"
            else
                print_warning "pipx installed but PATH not updated. Please restart your terminal or run:"
                echo "  export PATH=\"\$HOME/.local/bin:\$PATH\""
            fi
        else
            print_error "Failed to install pipx automatically"
            echo ""
            echo "Please install pipx manually:"
            echo "  python3 -m pip install --user pipx"
            echo "  python3 -m pipx ensurepath"
            echo ""
            echo "Then restart your terminal and run this installer again."
            exit 1
        fi
    fi
    
    # Install using pipx
    print_status "Installing proto-clickhouse-agent..."
    if ! pipx install proto-clickhouse-agent; then
        print_error "Failed to install Proto using pipx"
        print_warning "Trying alternative installation method..."
        
        # Fallback to pip install
        if python3 -m pip install --user proto-clickhouse-agent; then
            print_success "Proto installed successfully using pip!"
            print_status "Run 'proto' to start using it!"
            print_warning "If 'proto' command not found, restart your terminal or run:"
            echo "  export PATH=\"\$HOME/.local/bin:\$PATH\""
        else
            print_error "All installation methods failed"
            echo ""
            echo "Please try installing manually:"
            echo "  pipx install proto-clickhouse-agent"
            echo "  # or"
            echo "  pip install proto-clickhouse-agent"
            exit 1
        fi
    else
        print_success "Proto installed successfully!"
        print_status "Run 'proto' to start using it!"
        print_warning "First run will download the AI model (~3.5GB)"
    fi
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
        # Uninstall existing version
        pipx uninstall proto-clickhouse-agent 2>/dev/null || true
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
