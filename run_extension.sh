#!/bin/bash
cd "$(dirname "$0")"
echo "Starting YouTube Extension Server..."
./demucs.onnx/venv/bin/python src/extension/server/server.py
