#!/bin/bash
#
# ML Framework Setup Script
#
# This script automates the setup process for the ML Framework project.
# It checks for Poetry, installs dependencies, and verifies the installation.
#
# Usage:
#   bash scripts/setup.sh
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
print_header() {
    echo -e "${BLUE}================================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================================================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# Main setup function
main() {
    print_header "ML Framework Setup"
    
    # Check if we're in the project root
    if [ ! -f "pyproject.toml" ]; then
        print_error "pyproject.toml not found. Please run this script from the project root directory."
        echo "Usage: cd /path/to/ml-framework && bash scripts/setup.sh"
        exit 1
    fi
    
    print_success "Found project root directory"
    
    # Check Python version
    echo ""
    print_info "Checking Python version..."
    python_version=$(python3 --version 2>&1 | awk '{print $2}')
    print_success "Python $python_version found"
    
    # Check if Poetry is installed
    echo ""
    print_info "Checking for Poetry..."
    if ! command -v poetry &> /dev/null; then
        print_warning "Poetry is not installed"
        echo ""
        echo "Installing Poetry..."
        curl -sSL https://install.python-poetry.org | python3 -
        
        # Add Poetry to PATH
        export PATH="$HOME/.local/bin:$PATH"
        
        if command -v poetry &> /dev/null; then
            print_success "Poetry installed successfully"
        else
            print_error "Failed to install Poetry"
            echo "Please install Poetry manually from https://python-poetry.org/"
            exit 1
        fi
    else
        poetry_version=$(poetry --version)
        print_success "$poetry_version"
    fi
    
    # Install dependencies
    echo ""
    print_info "Installing project dependencies..."
    poetry install
    print_success "Dependencies installed"
    
    # Verify setup
    echo ""
    print_info "Verifying installation..."
    if poetry run python scripts/verify_setup.py; then
        echo ""
        print_header "Setup Complete!"
        echo ""
        print_success "Your ML Framework is ready to use!"
        echo ""
        echo "Next steps:"
        echo "  1. Activate the virtual environment:"
        echo "     poetry shell"
        echo ""
        echo "  2. Run a command:"
        echo "     poetry run python main.py --help"
        echo ""
        echo "  3. Or run scripts directly:"
        echo "     poetry run python scripts/entity_matching_pipeline.py --help"
        echo ""
        echo "For more information, see:"
        echo "  - docs/QUICKSTART.md"
        echo "  - docs/POETRY_SETUP.md"
        echo "  - README.md"
        echo ""
    else
        print_error "Verification failed. Please check the output above."
        exit 1
    fi
}

# Run main function
main
