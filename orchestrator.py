import requests
from TTS.api import TTS

# Step 1: Simulated or ASR-derived user input
user_input = "Tell me an interesting fact about space."

# Step 2: Call your local LLM server
LLM_API = "http://localhost:8000/v1/chat/completions"
llm_data = {"messages": [{"role": "user", "content": user_input}]}
llm_response = requests.post(LLM_API, json=llm_data).json()
llm_reply = llm_response["choices"][0]["message"]["content"]
print("LLM reply:", llm_reply)

# Step 3: Pass LLM reply directly to TTS for synthesis
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")
tts.tts_to_file(text=llm_reply, file_path="avatar_response.wav")
print("TTS synthesis complete. Output saved to avatar_response.wav")
