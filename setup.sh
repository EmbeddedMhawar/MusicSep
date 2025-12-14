#!/bin/bash
set -e

echo "=== MusicSep Setup ==="

# Check for Python 3
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 could not be found."
    exit 1
fi

# Check for ffmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo "Warning: ffmpeg could not be found. It is required for the YouTube Extension."
    echo "Please install it using your package manager (e.g., sudo pacman -S ffmpeg)"
fi

# Create venv
VENV_DIR="demucs.onnx/venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
else
    echo "Virtual environment already exists."
fi

# Install dependencies
echo "Installing dependencies..."
"$VENV_DIR/bin/pip" install --upgrade pip
"$VENV_DIR/bin/pip" install torch openunmix sounddevice numpy scipy fastapi uvicorn websockets yt-dlp

echo "=== Setup Complete ==="
echo "You can now run:"
echo "  ./run_desktop.sh    (for Desktop App)"
echo "  ./run_extension.sh  (for YouTube Extension)"
