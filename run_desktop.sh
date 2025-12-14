#!/bin/bash
cd "$(dirname "$0")"
echo "Starting Desktop Music Separator..."
./demucs.onnx/venv/bin/python src/desktop/main.py
