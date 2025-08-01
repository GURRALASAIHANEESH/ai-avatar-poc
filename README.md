# Real-Time AI Avatar System (WebSocket + Wav2Lip + TTS)

This project is a real-time, lip-synced AI avatar system that integrates:

* ChatGPT for conversational responses
* Text-to-Speech (TTS) for voice synthesis
* Wav2Lip for accurate facial animation
* Web frontend for user interaction
* WebSocket backend for real-time streaming

## 🔧 Project Structure

```
project-root/
│
├── client_web/                     # Frontend (static web client)
│   ├── index.html                  # Main UI with JavaScript & CSS
│   └── ...                         # Supporting JS, CSS, media assets
│
├── wav2lip_ws_server/             # Backend + ML pipeline (from Hugging Face)
│   └── gradio-lipsync-wav2lip/    # WebSocket + FastAPI backend
│       ├── server.py              # WebSocket server and FastAPI app
│       ├── inference.py           # Wav2Lip + TTS + processing logic
│       ├── windows_config.py      # Windows-specific compatibility patches
│       ├── audio.py               # Audio preprocessing and mel conversion
│       ├── file_manager.py        # Upload and manage video/audio files
│       ├── wav2lip_models/        # Wav2Lip model and checkpoint files
│       ├── face_detection/        # S3FD-based face detector
│       ├── requirements.txt       # Python dependencies
│       ├── .env.example           # Example env file
│       └── utils/                 # Utility modules (optional)
│
├── whisper.cpp/                  # (Optional) Offline STT module
│   ├── main, models, etc.
│
├── README.md
└── .gitignore
```

## 💡 Features

* ChatGPT integration for intelligent dialogue
* Text-to-Speech (TTS) pipeline
* Wav2Lip-based face animation
* Real-time WebSocket communication
* Simple web frontend for voice/text input
* Optional: Offline speech-to-text via `whisper.cpp`

## 📦 Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Install Python dependencies

```bash
cd wav2lip_ws_server/gradio-lipsync-wav2lip
pip install -r requirements.txt
```

### 3. Setup `.env`

Copy and fill in the environment config:

```bash
cp .env.example .env
```

### 4. Run Backend Server (WebSocket + FastAPI)

```bash
python server.py
```

The server will start on `ws://localhost:8001/ws/lipsync`

### 5. Open Frontend

Open `client_web/index.html` in your browser. This connects to the WebSocket backend and allows voice/text interaction.

## ✅ Windows-Specific Instructions

If you're on Windows, ensure `windows_config.py` is imported in `server.py` to patch system compatibility issues.

## 📁 Key Files and Directories

| Path                    | Description                          |
| ----------------------- | ------------------------------------ |
| `server.py`             | FastAPI + WebSocket server           |
| `inference.py`          | TTS + Wav2Lip logic                  |
| `windows_config.py`     | Windows compatibility patch          |
| `file_manager.py`       | Handles input/output file management |
| `audio.py`              | Audio processing for mel conversion  |
| `face_detection/`       | S3FD-based face detection            |
| `wav2lip_models/`       | Model weights and inference code     |
| `client_web/index.html` | Web frontend interface               |

## 🧪 Testing

* Input a message or voice from the browser.
* Backend streams TTS → lip-sync video → sends back final video.

## 🗂️ Optional Whisper Integration

* Add `whisper.cpp/` to use offline STT. Currently not integrated in pipeline.

## 🔐 .gitignore

Make sure the following are excluded:

```
*.pth
__pycache__/
*.mp4
*.wav
.env
```

## 🔄 Acknowledgments
* Hugging Face Spaces (for base backend)
* Gradio (initial interface version)

---

For questions or improvements, feel free to raise an issue or pull request.
