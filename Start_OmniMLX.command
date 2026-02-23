#!/bin/bash
# OmniMLX Launcher Script
# Double-click this file to start the OmniMLX Control Panel

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to the script directory
cd "$SCRIPT_DIR"

# Check if a virtual environment exists, if not create one
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Check if dependencies are installed
if ! python3 -c "import tkinter" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Run the application
echo "Starting OmniMLX Control Panel..."
python3 main.py

# Keep terminal open if there's an error
if [ $? -ne 0 ]; then
    echo ""
    echo "OmniMLX exited with an error. Press Enter to close."
    read
fi