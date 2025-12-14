#!/usr/bin/env python3
"""
Real-time audio separation using Open-Unmix (PyTorch).
Uses Blocking I/O to avoid Segfaults with PortAudio callbacks.
"""

import torch
import openunmix
import sounddevice as sd
import numpy as np
import threading
import queue
import time
import sys

# Restrict Torch threads
# torch.set_num_threads(1)

# Configuration
SAMPLE_RATE = 44100
CHUNK_DURATION = 1.0  # Seconds
STEM_NAMES = ["vocals", "drums", "bass", "other"]

class OpenUnmixRealtime:
    def __init__(self):
        print("Loading Open-Unmix (umxhq)...")
        self.model = openunmix.umxhq()
        self.model.eval()
        print("Model loaded!")
        
    def separate(self, audio_chunk):
        # audio_chunk: [2, N] numpy array
        input_tensor = torch.from_numpy(audio_chunk).float().unsqueeze(0)
        with torch.no_grad():
            output = self.model(input_tensor)
        output_np = output.squeeze(0).numpy() # [4, 2, N]
        stems = {}
        stems["vocals"] = output_np[0]
        stems["drums"] = output_np[1]
        stems["bass"] = output_np[2]
        stems["other"] = output_np[3]
        return stems

def main():
    print("=== Open-Unmix Real-Time Separation (Blocking I/O) ===\n")
    
    separator = OpenUnmixRealtime()
    
    # Queues
    input_queue = queue.Queue(maxsize=5)
    output_queue = queue.Queue(maxsize=5)
    
    # Flags
    running = True
    vocals_muted = False
    instrumental_muted = True
    
    # Device selection
    print("\nAvailable input devices:")
    devices = sd.query_devices()
    input_device = None
    for i, d in enumerate(devices):
        if d['max_input_channels'] > 0:
            print(f"  [{i}] {d['name']}")
            if 'monitor' in d['name'].lower() or 'music_input' in d['name'].lower():
                input_device = i
    
    if input_device is None:
        print("\nNo monitor device found. Please select input device number:")
        input_device = int(input())
    else:
        print(f"\nAuto-selected input: [{input_device}] {devices[input_device]['name']}")
        dev_info = devices[input_device]
        device_rate = int(dev_info['default_samplerate'])
        print(f"Device rate: {device_rate}")
        stream_rate = device_rate
    
    # Output Device selection
    print("\nAvailable output devices:")
    output_device = None
    for i, d in enumerate(devices):
        if d['max_output_channels'] > 0:
            print(f"  [{i}] {d['name']}")
    
    print("\nSelect output device number (default: system default):")
    output_rate = SAMPLE_RATE
    try:
        sel = input().strip()
        if sel:
            output_device = int(sel)
            print(f"Selected output: [{output_device}] {devices[output_device]['name']}")
            output_rate = int(devices[output_device]['default_samplerate'])
            print(f"Output device rate: {output_rate}")
        else:
            print("Using system default output.")
    except ValueError:
        print("Invalid input, using system default.")

    # Calculate buffer sizes
    INPUT_CHUNK_SAMPLES = int(CHUNK_DURATION * stream_rate)
    PROCESS_SAMPLES = int(CHUNK_DURATION * SAMPLE_RATE)
    
    print(f"Input chunk: {INPUT_CHUNK_SAMPLES} samples (@{stream_rate}Hz)")
    
    # --- Threads ---
    
    def read_loop():
        print("Read thread started")
        with sd.InputStream(samplerate=stream_rate, blocksize=4096, device=input_device, channels=2) as stream:
            buffer_list = []
            total_samples = 0
            
            while running:
                # Read small chunks to be responsive
                data, overflow = stream.read(4096)
                if overflow:
                    print("Input overflow")
                
                buffer_list.append(data) # data is [N, 2]
                total_samples += data.shape[0]
                
                if total_samples >= INPUT_CHUNK_SAMPLES:
                    # Concatenate
                    full_chunk = np.concatenate(buffer_list, axis=0)
                    # Keep exact size
                    chunk_to_process = full_chunk[:INPUT_CHUNK_SAMPLES]
                    
                    # Handle remainder
                    remainder = full_chunk[INPUT_CHUNK_SAMPLES:]
                    buffer_list = [remainder]
                    total_samples = remainder.shape[0]
                    
                    # Put in queue (transpose to [2, N] for processing)
                    try:
                        input_queue.put(chunk_to_process.T, timeout=1.0)
                    except queue.Full:
                        pass

    def process_loop():
        print("Process thread started")
        while running:
            try:
                audio = input_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            
            # Resample if needed
            if stream_rate != SAMPLE_RATE:
                from scipy.signal import resample
                new_samples = int(audio.shape[1] * SAMPLE_RATE / stream_rate)
                audio = resample(audio, new_samples, axis=1)
            
            # Calculate Input RMS
            input_rms = np.sqrt(np.mean(audio**2))
            
            start = time.time()
            stems = separator.separate(audio)
            elapsed = time.time() - start
            print(f"Proc: {elapsed:.2f}s ({CHUNK_DURATION/elapsed:.1f}x) | Input RMS: {input_rms:.4f}")
            
            # Mix
            vocals = stems["vocals"]
            instrumental = stems["drums"] + stems["bass"] + stems["other"]
            
            # Mute logic
            output_mix = np.zeros_like(vocals)
            if not vocals_muted:
                output_mix += vocals
            if not instrumental_muted:
                output_mix += instrumental
            
            # Put in output queue (transpose back to [N, 2] for playback)
            try:
                output_queue.put(output_mix.T, timeout=1.0)
            except queue.Full:
                pass

    def write_loop():
        print("Write thread started")
        # Output at device rate
        with sd.OutputStream(samplerate=output_rate, blocksize=4096, device=output_device, channels=2) as stream:
            current_chunk = None
            cursor = 0
            
            while running:
                if current_chunk is None:
                    try:
                        data = output_queue.get(timeout=0.1)
                        
                        # Resample output if needed (44100 -> output_rate)
                        if output_rate != SAMPLE_RATE:
                            from scipy.signal import resample
                            new_samples = int(data.shape[0] * output_rate / SAMPLE_RATE)
                            data = resample(data, new_samples, axis=0)
                            
                        current_chunk = data
                        cursor = 0
                    except queue.Empty:
                        # Write silence if no data
                        silence = np.zeros((1024, 2), dtype=np.float32)
                        stream.write(silence)
                        continue
                
                # Write in small blocks to keep stream alive
                remaining = current_chunk.shape[0] - cursor
                to_write = min(remaining, 4096)
                
                chunk_part = current_chunk[cursor:cursor+to_write]
                
                # Verify contiguous
                if not chunk_part.flags['C_CONTIGUOUS']:
                    chunk_part = np.ascontiguousarray(chunk_part)
                
                stream.write(chunk_part)
                cursor += to_write
                
                if cursor >= current_chunk.shape[0]:
                    current_chunk = None

    # Start threads
    t_read = threading.Thread(target=read_loop, daemon=True)
    t_proc = threading.Thread(target=process_loop, daemon=True)
    t_write = threading.Thread(target=write_loop, daemon=True)
    
    t_read.start()
    t_proc.start()
    t_write.start()
    
    print("Streaming... Press V/I to toggle, Q to quit.")
    
    try:
        while True:
            cmd = input().strip().lower()
            if cmd == 'v':
                vocals_muted = not vocals_muted
                print(f"Vocals: {'MUTED' if vocals_muted else 'ON'}")
            elif cmd == 'i':
                instrumental_muted = not instrumental_muted
                print(f"Instrumental: {'MUTED' if instrumental_muted else 'ON'}")
            elif cmd == 'q':
                break
    except KeyboardInterrupt:
        pass
    
    running = False
    print("Stopping...")
    time.sleep(1) # Give threads time to exit

if __name__ == "__main__":
    main()
