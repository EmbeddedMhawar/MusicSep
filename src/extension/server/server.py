import asyncio
import subprocess
import threading
import queue
import time
import numpy as np
import torch
import openunmix
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import sys
import os

# --- Configuration ---
SAMPLE_RATE = 44100
CHUNK_DURATION = 1.0  # Seconds (Look-ahead chunk)
CHUNK_SAMPLES = int(CHUNK_DURATION * SAMPLE_RATE)

# --- Open-Unmix Model ---
print("Loading Open-Unmix (umxhq)...")
# Force single thread for torch to avoid contention
torch.set_num_threads(1)
separator = openunmix.umxhq()
separator.eval()
print("Model loaded!")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global State ---
class AudioState:
    def __init__(self):
        self.process = None
        self.audio_queue = queue.Queue(maxsize=10)
        self.output_queue = queue.Queue(maxsize=10)
        self.running = False
        self.current_url = ""

state = AudioState()

class StartRequest(BaseModel):
    url: str
    timestamp: float

def separate_audio(audio_chunk):
    # audio_chunk: [2, N] numpy array
    input_tensor = torch.from_numpy(audio_chunk).float().unsqueeze(0)
    with torch.no_grad():
        output = separator(input_tensor)
    output_np = output.squeeze(0).numpy() # [4, 2, N]
    stems = {}
    stems["vocals"] = output_np[0]
    stems["drums"] = output_np[1]
    stems["bass"] = output_np[2]
    stems["other"] = output_np[3]
    return stems

def processing_loop():
    print("Processing thread started")
    buffer_list = []
    total_samples = 0
    
    while state.running:
        try:
            # Get raw PCM data (bytes)
            chunk_bytes = state.audio_queue.get(timeout=1.0)
        except queue.Empty:
            continue
            
        # Convert bytes to numpy (float32, stereo)
        audio_chunk = np.frombuffer(chunk_bytes, dtype=np.float32)
        # Reshape to [N, 2]
        if len(audio_chunk) % 2 != 0:
            continue # Incomplete frame
        audio_chunk = audio_chunk.reshape(-1, 2)
        
        buffer_list.append(audio_chunk)
        total_samples += audio_chunk.shape[0]
        
        if total_samples >= CHUNK_SAMPLES:
            # Concatenate
            full_chunk = np.concatenate(buffer_list, axis=0)
            
            # Process strictly CHUNK_SAMPLES
            to_process = full_chunk[:CHUNK_SAMPLES]
            remainder = full_chunk[CHUNK_SAMPLES:]
            
            buffer_list = [remainder]
            total_samples = remainder.shape[0]
            
            # Transpose to [2, N] for model
            to_process_T = to_process.T
            
            # Separate
            stems = separate_audio(to_process_T)
            
            # Mix (Vocals vs Instrumental)
            # We send both? Or just mix based on request?
            # For simplicity, let's send Vocals (Left) and Instrumental (Right) 
            # so client can mix with balance? 
            # Or better: Send interleaved stereo of the MIX.
            # Let's assume client wants Vocals by default for now, 
            # but we can implement a toggle later.
            # Actually, let's send Vocals + Instrumental as 4 channels? 
            # WebSocket binary is simple. Let's send just Vocals for now to test.
            
            # Better: Send Vocals (Stereo).
            vocals = stems["vocals"] # [2, N]
            
            # Transpose back to [N, 2]
            vocals_out = vocals.T
            
            # Convert to bytes
            out_bytes = vocals_out.astype(np.float32).tobytes()
            
            try:
                state.output_queue.put(out_bytes, timeout=1.0)
            except queue.Full:
                pass

def stream_downloader(url, start_time):
    print(f"Starting download: {url} @ {start_time}s")
    
    # Command to stream PCM float32 le 44100Hz stereo
    # yt-dlp -> stdout -> ffmpeg -> stdout
    
    # 1. yt-dlp command
    # Note: -g gets the URL, but we want the stream.
    # We use -o - to pipe.
    
    # We need to find the direct URL first to seek properly with ffmpeg?
    # Or can yt-dlp seek? --download-sections "*10:20"
    # --begin is deprecated? No, use --download-sections.
    # Format: *START_TIME-inf
    section = f"*{start_time}-inf"
    
    # Absolute path to yt-dlp in venv
    # We are in src/extension/server/server.py
    # Venv is in ../../../demucs.onnx/venv
    # But better to use absolute path or relative from CWD if run from root.
    # Let's use absolute path for safety.
    yt_dlp_path = "/home/mhawar/Desktop/Side Project/MusicSep/demucs.onnx/venv/bin/yt-dlp"
    
    cmd = [
        yt_dlp_path,
        "-f", "bestaudio",
        "--download-sections", section,
        "-o", "-",
        "-q", # quiet
        url
    ]
    
    # Pipe to ffmpeg to ensure f32le 44100
    ffmpeg_cmd = [
        "ffmpeg",
        "-i", "pipe:0",
        "-f", "f32le",
        "-ac", "2",
        "-ar", "44100",
        "pipe:1"
    ]
    
    p1 = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p2 = subprocess.Popen(ffmpeg_cmd, stdin=p1.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    state.process = p2
    
    # Read loop
    chunk_size = 4096 * 2 * 4 # 4096 frames * 2 channels * 4 bytes
    
    while state.running:
        data = p2.stdout.read(chunk_size)
        if not data:
            break
        state.audio_queue.put(data)
        
    print("Download finished/stopped")
    p2.terminate()
    p1.terminate()

@app.post("/start")
async def start_stream(req: StartRequest):
    # Stop existing
    state.running = False
    if state.process:
        state.process.terminate()
    
    # Clear queues
    with state.audio_queue.mutex:
        state.audio_queue.queue.clear()
    with state.output_queue.mutex:
        state.output_queue.queue.clear()
        
    time.sleep(0.5) # Wait for threads
    
    state.running = True
    state.current_url = req.url
    
    # Start threads
    t_dl = threading.Thread(target=stream_downloader, args=(req.url, req.timestamp), daemon=True)
    t_proc = threading.Thread(target=processing_loop, daemon=True)
    
    t_dl.start()
    t_proc.start()
    
    return {"status": "started"}

@app.websocket("/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            if not state.output_queue.empty():
                data = state.output_queue.get()
                await websocket.send_bytes(data)
            else:
                await asyncio.sleep(0.01)
    except WebSocketDisconnect:
        print("Client disconnected")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
