import requests

# Instead of calling ASR, just use a string directly:
asr_transcript = "What is artificial intelligence and how does it work?"

LLAMA_API_URL = "http://localhost:8000/v1/chat/completions"
data = {
    "messages": [{"role": "user", "content": asr_transcript}]
}

resp = requests.post(LLAMA_API_URL, json=data)
reply = resp.json()["choices"][0]["message"]["content"]

print("LLM reply:", reply)
