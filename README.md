# 🧠 Real-Time AI Avatar Lip Sync System

This project is a real-time, end-to-end AI avatar pipeline that allows users to input speech or text via a web frontend, generates smart responses using OpenAI's ChatGPT API, converts them to speech (TTS), and synchronizes the audio onto a talking avatar using Wav2Lip. All communication is handled through an optimized WebSocket protocol for real-time interaction.

---

## 📁 Project Structure

```
project-root/
│
├── client_web/                 # Frontend (static web client)
│   ├── index.html              # Main UI with JavaScript & CSS
│   ├── ...                     # Supporting JS, CSS, assets (optional)
│
├── wav2lip_ws_server/
│   └── gradio-lipsync-wav2lip/ # Backend + ML pipeline (from Hugging Face)
│       ├── server.py           # WebSocket + FastAPI backend
│       ├── inference.py        # Wav2Lip + TTS + processing logic
│       ├── requirements.txt    # Python dependencies
│       ├── .env.example        # Example env file
│       ├── face_detection/     # Face detection module for Wav2Lip
│       └── wav2lip.pth         # Pretrained Wav2Lip model
│
├── whisper.cpp/               # (Optional) For offline speech-to-text
│   ├── main, models, etc.
│
├── README.md                  # This file
└── .gitignore                 # Ignore large/sensitive files
```

---

## 🌐 Frontend

**Location:** `client_web/`

* `index.html`: Interactive UI for voice and text input
* WebSocket client for communicating with backend
* Uses Web Speech API, TTS, and MediaRecorder APIs

**How to Run:**

```bash
cd client_web
python -m http.server 8080
```

Visit `http://localhost:8080` in browser.

---

## 🚀 Backend

**Location:** `wav2lip_ws_server/gradio-lipsync-wav2lip/`

Handles:

* WebSocket binary message parsing
* ChatGPT interaction
* OpenAI TTS
* Wav2Lip inference
* Returns lip-synced avatar video

### 📦 Backend Setup:

1. Clone the base repository:

```bash
git clone https://huggingface.co/spaces/banao-tech/gradio-lipsync-wav2lip
```

2. Modify the contents:

* Replace `inference.py` with your custom logic
* Add `server.py` (FastAPI + WebSocket backend)
* Ensure `face_detection/` and `wav2lip.pth` are present

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set environment variables:

```bash
cp .env.example .env
# Edit with your API keys
```

5. Start server:

```bash
python server.py
```

Visit backend at `http://localhost:8001`

---

## 🔊 Whisper.cpp (Optional for STT)

To enable offline speech-to-text:

```bash
git clone https://github.com/ggerganov/whisper.cpp
cd whisper.cpp
make
```

Use it in place of browser STT if needed.

---

## 🧠 Dependencies

### Python (backend)

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

### Environment

`.env.example`

```env
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
HF_TOKEN=hf-xxxxxxxxxxxxxxxxxxxxxxxx
```

---

## 🧹 .gitignore

```
*.pth
*.mp4
*.wav
*.avi
*.h5
venv/
__pycache__/
.env
.DS_Store
```

---

## 🧪 Testing

* Test with text input: should generate avatar video with synced lips
* Test with voice input: ensure correct audio is sent and processed
* Watch WebSocket logs for parsing and output video timing

---

## 🛠️ Known Issues

* Voice input sometimes doesn't return video (debug audio format)
* Lip sync quality may vary by avatar resolution and padding
* 10–15s average latency per response

---

## 📌 Future Enhancements

* Improve voice input stability
* Add emotional tone control in TTS
* Optimize avatar resolution and padding for lip sync accuracy
* Add multiple avatar support
* Deploy on cloud with streaming support

---

## 🤝 Support

For issues, create a GitHub issue or contact the maintainer.

