# AI Avatar Lip Sync System

This project integrates ChatGPT responses with a real-time lip-sync video generation system. It allows users to interact with an AI assistant via text or voice, with the avatar responding in synchronized speech and lip movements.

## Features

* Real-time AI responses (ChatGPT/OpenAI)
* Text-to-Speech (TTS) conversion
* Lip-sync video generation using Wav2Lip
* Audio chunk streaming to reduce latency
* Model warm-up for faster response times
* File manager to upload video clips for lip sync
* WebSocket support for real-time communication

## Tech Stack

* **Backend**: FastAPI, WebSockets, Wav2Lip, PyTorch
* **Frontend**: HTML, JavaScript, WebSocket, MediaRecorder
* **Models**: Wav2Lip, S3FD Face Detector
* **TTS**: Edge-TTS (can be swapped)

---

## Directory Structure

```
├── app
│   ├── main.py                # FastAPI backend with WebSocket support
│   ├── windows_config.py      # Windows-specific model and path configurations
│   ├── wav2lip_models         # Wav2Lip model definitions
│   ├── face_detection         # Face detector using S3FD
│   ├── audio.py               # Audio preprocessing and mel-spectrogram conversion
│   └── file_manager.py        # File handling and video uploads
├── static
│   ├── index.html             # Frontend interface
│   └── ...                    # Other static assets
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/ai-avatar-lipsync.git
cd ai-avatar-lipsync
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Make sure you have:

* Python 3.8+
* FFmpeg installed and in PATH
* `torch` with CUDA support (if using GPU)

### 3. Download Pretrained Models

Place the following files:

* `wav2lip.pth` in `app/wav2lip_models/checkpoints`
* `s3fd.pth` in `app/face_detection/detection/sfd`

Download links (official):

* Wav2Lip model: [https://github.com/Rudrabha/Wav2Lip](https://github.com/Rudrabha/Wav2Lip)
* S3FD Face Detector: [https://github.com/clcarwin/s3fd](https://github.com/clcarwin/s3fd)

---

## Running the Application

### Start Backend Server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8001
```

### Open Frontend

Open `static/index.html` in a browser. Use microphone or text to start a conversation.

---

## How It Works

1. User speaks or types a message.
2. Message is sent to backend via WebSocket.
3. Backend performs:

   * Transcription (if voice)
   * AI response (ChatGPT)
   * TTS conversion
   * Streaming audio chunks to Wav2Lip for lip sync
4. Final video is sent back to frontend.

### Wav2Lip Pipeline (Optimized)

* **Model Preloading**: All models (face detector, encoders, decoder) are loaded once at server startup.
* **Chunk-based Audio Handling**: TTS audio is streamed in chunks to speech encoder and synced with video.

---

## Windows Notes

* `windows_config.py` includes special fixes for paths, subprocess handling, and multiprocessing issues on Windows.
* Ensure correct `ffmpeg` path is set if not in system path.

---

## File Manager Support

* Upload a video clip using the file manager UI.
* The backend will apply lip-sync generation using TTS audio or uploaded speech.

---

## TODO

* Add support for GPU batch inference
* Improve latency using async video encoding
* Add avatar face overlay with chroma-key blending
* Deploy with Docker (optional)

---

## License

MIT License

---

##
