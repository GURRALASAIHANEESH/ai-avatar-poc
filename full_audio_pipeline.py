import os
from dotenv import load_dotenv

# Load .env file from project root
load_dotenv()

import subprocess
import sys
import requests
from TTS.api import TTS
import pyaudio
import wave
from wav2lip_cloud import generate_lipsync_video

# --- CONFIGURATION ---
RECORD_SECONDS = 5
INPUT_WAV    = "D:/AI-RealTime-POC/input.wav"
OUTPUT_TXT   = "D:/AI-RealTime-POC/input.wav.txt"
AVATAR_VIDEO = "D:/AI-RealTime-POC/client_web/avatar_sample_green.mp4"  # or .png
OUTPUT_DIR   = "D:/AI-RealTime-POC/client_web"
TTS_WAV      = "D:/AI-RealTime-POC/avatar_response.wav"
WHISPER_EXE  = r"D:\AI-RealTime-POC\asr_whisper\whisper.cpp\build\bin\Release\whisper-cli.exe"
MODEL_PATH   = r"D:\AI-RealTime-POC\asr_whisper\whisper.cpp\models\ggml-base.en.bin"
LLM_API_URL  = "http://localhost:8000/v1/chat/completions"
hf_token     = os.getenv("HF_TOKEN")

# --- Step 1: Audio Capture ---
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024

audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
print("Recording from mic...")
frames = [stream.read(CHUNK) for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS))]
print("Recorded.")
stream.stop_stream()
stream.close()
audio.terminate()

with wave.open(INPUT_WAV, 'wb') as wf:
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))

# --- Step 2: Transcribe ---
print("Transcribing with whisper.cpp...")
subprocess.run([WHISPER_EXE, "-m", MODEL_PATH, "-f", INPUT_WAV, "-otxt"], check=True)

with open(OUTPUT_TXT, 'r', encoding='utf-8') as f:
    transcript = f.read().strip()
print(f"ASR Transcript: {transcript}")

# --- Step 3: LLM (local API) ---
data = {"messages": [{"role": "user", "content": transcript}]}
try:
    llm_resp = requests.post(LLM_API_URL, json=data)
    llm_resp.raise_for_status()
    llm_text = llm_resp.json()["choices"][0]["message"]["content"]
except Exception as e:
    print(f"LLM API call failed: {e}")
    sys.exit(1)
print(f"LLM Reply: {llm_text}")

# --- Step 4: TTS (Coqui) ---
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")
tts.tts_to_file(text=llm_text, file_path=TTS_WAV)
print(f"TTS synthesis complete. Output in {TTS_WAV}")

# --- Step 5: Lipsync (Cloud Wav2Lip) ---
try:
    lipsynced_video = generate_lipsync_video(
        video_path=AVATAR_VIDEO,
        audio_path=TTS_WAV,
        output_dir=OUTPUT_DIR,
        hf_token=hf_token
    )
    print(f"Lipsynced video generated successfully at {lipsynced_video}")
except Exception as e:
    print(f"Wav2Lip cloud inference failed: {e}")
    sys.exit(1)
