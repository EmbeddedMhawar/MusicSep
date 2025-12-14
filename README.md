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
1.  Clone the repo:
    ```bash
    git clone https://github.com/EmbeddedMhawar/MusicSep.git
    cd MusicSep
    ```
2.  Run the setup script:
    ```bash
    ./setup.sh
    ```
    This will create the virtual environment and install all dependencies.
