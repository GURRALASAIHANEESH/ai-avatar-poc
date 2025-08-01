Here is your updated `README.md` with all recent changes, including the addition of `file_manager.py` and `windows_config.py`, removal of all emojis, and improved formatting for clarity and professionalism:

```markdown
# Real-Time AI Avatar Lip Sync System

This project is a real-time, end-to-end AI avatar pipeline that allows users to input speech or text via a web frontend, generate smart responses using OpenAI's ChatGPT API, convert them to speech using OpenAI TTS, and synchronize the audio onto a talking avatar using Wav2Lip. All communication is handled through an optimized WebSocket protocol for near real-time interaction.

---

## Project Structure

```

project-root/
│
├── client\_web/                   # Frontend (static web client)
│   ├── index.html                # Main UI with JavaScript & CSS
│   └── ...                       # Supporting JS, CSS, assets
│
├── wav2lip\_ws\_server/
│   └── gradio-lipsync-wav2lip/   # Backend + ML pipeline
│       ├── server.py             # WebSocket + FastAPI backend
│       ├── inference.py          # Wav2Lip + TTS + response logic
│       ├── file\_manager.py       # Utility for file chunking and sync
│       ├── windows\_config.py     # Optional system cleanup script (Windows only)
│       ├── requirements.txt      # Python dependencies
│       ├── .env.example          # Example environment config
│       ├── face\_detection/       # Wav2Lip face detection module
│       └── wav2lip.pth           # Pretrained model weights
│
├── whisper.cpp/                 # (Optional) Offline STT engine
│   ├── main, models, etc.
│
├── README.md                    # Project documentation
└── .gitignore                   # Ignore large or sensitive files

````

---

## Frontend

**Location:** `client_web/`

The frontend consists of a minimal HTML/JavaScript interface that captures user voice or text input and communicates with the backend via WebSocket.

- `index.html` provides the interactive UI
- Uses Web APIs (Web Speech API, MediaRecorder, AudioContext)
- Sends text/audio to backend and renders received video

**How to Run:**

```bash
cd client_web
python -m http.server 8080
````

Open your browser and navigate to:
`http://localhost:8080`

---

## Backend

**Location:** `wav2lip_ws_server/gradio-lipsync-wav2lip/`

Responsible for:

* Parsing and handling WebSocket connections
* Communicating with OpenAI GPT and TTS APIs
* Generating lip-synced video using Wav2Lip
* Chunking responses and streaming to frontend

### Backend Setup

1. Clone the base project or Hugging Face Space:

```bash
git clone https://huggingface.co/spaces/banao-tech/gradio-lipsync-wav2lip
```

2. Add or replace files:

* Replace `inference.py` with custom logic
* Add `server.py` for FastAPI + WebSocket support
* Add `file_manager.py` to support chunk-based streaming
* Add `windows_config.py` (optional utility for developers)

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set environment variables:

```bash
cp .env.example .env
# Then edit .env with your keys
```

5. Start backend server:

```bash
python server.py
```

Access it at:
`http://localhost:8001`

---

## Windows Utilities

**File:** `windows_config.py`
This is a Windows-specific utility script to clean up temporary files, cache folders, and unused models. It supports parallel deletion, logging, dry-run mode, and orphaned pip package cleanup.

Run it manually as needed:

```bash
python windows_config.py --delete-models --delete-orphan-pip --log cleanup.txt
```

Useful for developers working with large models, environments, or long sessions on Windows.

---

## File Streaming

**File:** `file_manager.py`
Used by the backend to:

* Split WAV and video output into chunks
* Enable real-time streaming of audio/video to frontend
* Smoothly deliver output over WebSocket while it's being processed

---

## Whisper.cpp (Optional STT)

To use offline transcription:

```bash
git clone https://github.com/ggerganov/whisper.cpp
cd whisper.cpp
make
```

Can be used in place of browser’s speech recognition for voice input processing.

---

## Python Dependencies

Listed in `requirements.txt`. Core libraries include:

```
fastapi
uvicorn
websockets
python-dotenv
torch
torchvision
torchaudio
librosa
numpy
opencv-python
aiofiles
aiohttp
soundfile
requests
```

---

## Environment File

`.env.example`

```env
OPENAI_API_KEY=your_openai_api_key
HF_TOKEN=your_huggingface_token
```

---

## .gitignore

```
*.pth
*.mp4
*.wav
*.avi
*.h5
*.log
venv/
__pycache__/
.env
.DS_Store
```

---

## Testing

* Use text input to trigger GPT + TTS + video generation
* Use voice input via browser mic to test end-to-end pipeline
* Confirm streaming video is rendered on frontend
* Check WebSocket logs for chunk sync and timing

---

## Known Issues

* Audio input with high latency or silence may break flow
* Lip sync quality depends on avatar resolution and preprocessing
* Video latency ranges from 10 to 15 seconds on average
* Limited browser support for MediaRecorder API on iOS

---

## Future Enhancements

* Improve WebSocket audio buffer handling
* Add avatar emotion control via GPT prompt injection
* Integrate more TTS voices with expressiveness
* Support for avatars with different face geometries
* Deploy backend on GPU-supported cloud server for scalability

---

## Support

For issues, contributions, or bugs, open a GitHub issue or contact the project maintainer.

```

Let me know if you'd like this saved as a file or converted into a `README.txt` version for internal docs.
```
