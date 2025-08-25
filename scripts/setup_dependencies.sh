#!/bin/bash
# AI-AGENT-REF: System dependency setup script for AI Trading Bot

set -e

echo "Setting up system dependencies for AI Trading Bot..."

# Detect the operating system
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Detected Linux system"
    
    # Update package list
    echo "Updating package list..."
    sudo apt-get update
    
    # Setup TA-Lib system dependencies
    echo "Setting up TA-Lib system dependencies..."
    sudo apt-get install -y libta-lib0-dev ta-lib-common
    
    # Install other useful dependencies
    echo "Installing additional system dependencies..."
    sudo apt-get install -y \
        build-essential \
        python3-dev \
        python3-pip \
        git \
        curl \
        wget
        
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Detected macOS system"
    
    # Check if Homebrew is installed
    if ! command -v brew &> /dev/null; then
        echo "Homebrew not found. Please install Homebrew first:"
        echo '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
        exit 1
    fi
    
    # Setup TA-Lib via Homebrew
    echo "Setting up TA-Lib via Homebrew..."
    brew install 'ta-lib'
    
else
    echo "Warning: Unsupported operating system: $OSTYPE"
    echo "Please ensure TA-Lib system dependencies are present:"
    echo "- On Ubuntu/Debian: sudo apt-get install libta-lib0-dev"
    echo "- On macOS: brew install 'ta-lib'"
    echo "- On other systems: see https://github.com/mrjbq7/ta-lib#dependencies"
fi

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Verify TA-Lib installation
echo "Verifying TA-Lib installation..."
python -c "import talib; print('TA-Lib successfully installed and working!')" || {
    echo "WARNING: TA-Lib Python package installation failed."
    echo "The system will use fallback implementations."
    echo "For enhanced technical analysis, ensure TA-Lib is properly installed:"
    echo "- System library: libta-lib0-dev (Linux) or 'ta-lib' (macOS)"
    echo "- Python package: pip install \"ai-trading-bot[ta]\""
}

echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Copy .env.example to .env and configure your API keys"
echo "2. Run the trading bot with: python run.py"
echo "3. Check logs/ directory for execution logs"