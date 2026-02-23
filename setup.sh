#!/bin/bash
#
# OmniMLX Setup Script
# This script sets up the virtual environment and creates a launcher for easy access.
#

set -e

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "╔════════════════════════════════════════════════════════════╗"
echo "║              OmniMLX Setup Script v1.0                     ║"
echo "║     The All-in-One Local AI Server Manager                 ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check for uv
echo "▶ Checking for uv package manager..."
if ! command -v uv &> /dev/null; then
    echo "  uv not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add uv to PATH for this session
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "  ✓ uv is installed"
fi

# Create virtual environment
echo ""
echo "▶ Creating virtual environment..."
if [ ! -d ".venv" ]; then
    uv venv
    echo "  ✓ Virtual environment created"
else
    echo "  ✓ Virtual environment already exists"
fi

# Install dependencies
echo ""
echo "▶ Installing dependencies..."
uv pip install -r requirements.txt
echo "  ✓ Dependencies installed"

# Generate the launcher script
echo ""
echo "▶ Generating OmniMLX_Launcher.command..."

LAUNCHER_PATH="$SCRIPT_DIR/OmniMLX_Launcher.command"

cat > "$LAUNCHER_PATH" << 'LAUNCHER_EOF'
#!/bin/bash
#
# OmniMLX Launcher
# Double-click this file to launch OmniMLX
#

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if venv exists
if [ ! -d ".venv" ]; then
    echo "Error: Virtual environment not found!"
    echo "Please run ./setup.sh first"
    read -p "Press Enter to close..."
    exit 1
fi

# Activate virtual environment and run
source .venv/bin/activate
python main.py

# Keep window open if there's an error
if [ $? -ne 0 ]; then
    echo ""
    echo "OmniMLX exited with an error."
    read -p "Press Enter to close..."
fi
LAUNCHER_EOF

# Make launcher executable
chmod +x "$LAUNCHER_PATH"
echo "  ✓ Launcher created: OmniMLX_Launcher.command"

# Final message
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                    Setup Complete!                         ║"
echo "╠════════════════════════════════════════════════════════════╣"
echo "║                                                            ║"
echo "║  To launch OmniMLX:                                       ║"
echo "║    Double-click: OmniMLX_Launcher.command                 ║"
echo "║                                                            ║"
echo "║  Or from terminal:                                        ║"
echo "║    source .venv/bin/activate && python main.py            ║"
echo "║                                                            ║"
echo "╚════════════════════════════════════════════════════════════╝"