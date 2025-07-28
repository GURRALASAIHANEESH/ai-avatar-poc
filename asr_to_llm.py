# File: pipeline_scripts/asr_to_llm.py

import requests

# 1. Read the transcript from Whisper ASR
with open("D:/ai-realtime-avatar-poc/input.wav.txt", "r", encoding="utf-8") as f:
    transcript = f.read().strip()
print("ASR Transcript:", transcript)

# 2. Send ASR transcript to your running local LLM server (adjust port if needed)
LLM_API_URL = "http://localhost:8000/v1/chat/completions"
data = {
    "messages": [{"role": "user", "content": transcript}]
}

response = requests.post(LLM_API_URL, json=data)
reply = response.json()["choices"][0]["message"]["content"]
print("LLM reply:", reply)
