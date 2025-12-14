# MusicSep (Work in Progress)

Real-time music separation on Linux using Open-Unmix (PyTorch).

## Features
*   **Desktop App**: Real-time separation from system audio (via Virtual Cable).
    *   Status: **Working**.
    *   Run: `./run_desktop.sh`
*   **YouTube Extension**: Zero-latency separation for YouTube videos (Look-Ahead).
    *   Status: **In Progress** (Syncing issues).
    *   Run Server: `./run_extension.sh`
    *   Load Extension: `src/extension/extension`

## Requirements
*   Python 3.13+
*   `ffmpeg`
*   `virtual-cable` (for Desktop App)

## Installation
1.  Clone the repo.
2.  The `demucs.onnx/venv` contains the environment (not included in repo, needs setup).
    *   *Note: This repo currently assumes a local venv setup.*
