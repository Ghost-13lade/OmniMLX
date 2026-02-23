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
