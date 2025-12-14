let audioCtx;
let ws;
let nextStartTime = 0;
let isSeparating = false;

// Listen for messages from popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "start") {
        startSeparation();
    } else if (request.action === "stop") {
        stopSeparation();
    }
});

async function startSeparation() {
    const video = document.querySelector('video');
    if (!video) return;

    isSeparating = true;
    video.muted = true; // Mute original
    console.log("Starting separation...");

    // Init AudioContext
    if (!audioCtx) {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 44100 });
    }

    // Connect WebSocket
    ws = new WebSocket("ws://localhost:8000/stream");
    ws.binaryType = "arraybuffer";

    ws.onmessage = async (event) => {
        if (!isSeparating) return;

        // Decode Float32 PCM
        const pcmData = new Float32Array(event.data);

        // Create AudioBuffer
        const buffer = audioCtx.createBuffer(2, pcmData.length / 2, 44100);
        const ch0 = buffer.getChannelData(0);
        const ch1 = buffer.getChannelData(1);

        // De-interleave (L R L R...)
        for (let i = 0; i < pcmData.length / 2; i++) {
            ch0[i] = pcmData[i * 2];
            ch1[i] = pcmData[i * 2 + 1];
        }

        // Schedule playback
        const source = audioCtx.createBufferSource();
        source.buffer = buffer;
        source.connect(audioCtx.destination);

        if (nextStartTime < audioCtx.currentTime) {
            nextStartTime = audioCtx.currentTime;
        }
        source.start(nextStartTime);
        nextStartTime += buffer.duration;
    };

    // Tell backend to start downloading
    const url = window.location.href;
    const timestamp = video.currentTime;

    await fetch("http://localhost:8000/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: url, timestamp: timestamp })
    });

    // Handle Seeking
    video.onseeked = async () => {
        if (!isSeparating) return;
        console.log("Seek detected! Restarting backend...");

        // Stop current audio queue
        if (audioCtx) {
            await audioCtx.close();
            audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 44100 });
            nextStartTime = 0;
        }

        // Restart backend
        await fetch("http://localhost:8000/start", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ url: window.location.href, timestamp: video.currentTime })
        });
    };
}

function stopSeparation() {
    isSeparating = false;
    const video = document.querySelector('video');
    if (video) video.muted = false;
    if (ws) ws.close();
    if (audioCtx) audioCtx.close();
    audioCtx = null;
    console.log("Stopped separation.");
}
