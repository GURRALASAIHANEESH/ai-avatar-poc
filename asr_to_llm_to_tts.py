import requests
from TTS.api import TTS

# 1. Read transcript
with open("D:/ai-realtime-avatar-poc/input.wav.txt", "r", encoding="utf-8") as f:
    transcript = f.read().strip()

# 2. Get LLM reply
LLM_API_URL = "http://localhost:8000/v1/chat/completions"
data = {"messages": [{"role": "user", "content": transcript}]}
response = requests.post(LLM_API_URL, json=data)
reply = response.json()["choices"][0]["message"]["content"]

print("ASR Transcript:", transcript)
print("LLM reply:", reply)

# 3. Convert LLM reply to speech with TTS
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")  # or your preferred model
tts.tts_to_file(text=reply, file_path="avatar_response.wav")
print("TTS synthesis complete. Output saved to avatar_response.wav")
