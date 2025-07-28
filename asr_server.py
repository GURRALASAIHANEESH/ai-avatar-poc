from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import subprocess
import uuid
import os

app = FastAPI()

WHISPER_EXECUTABLE = r"D:\ai-realtime-avatar-poc\asr_whisper\whisper.cpp\build\bin\Release\whisper-cli.exe"
MODEL_PATH = r"D:\ai-realtime-avatar-poc\asr_whisper\whisper.cpp\ggml-base.en.bin"

@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    try:
        tmp_filename = f"temp_{uuid.uuid4()}.wav"
        with open(tmp_filename, "wb") as f:
            f.write(await audio.read())

        # Call whisper.cpp binary
        result = subprocess.run(
            [WHISPER_EXECUTABLE, "-m", MODEL_PATH, "-f", tmp_filename, "-otxt", "-of", tmp_filename],
            capture_output=True, text=True
        )

        if result.returncode != 0:
            return JSONResponse(status_code=500, content={
                "error": f"Whisper execution failed: {result.stderr}"
            })

        transcript_file = tmp_filename + ".txt"
        if not os.path.exists(transcript_file):
            return JSONResponse(status_code=500, content={"error": "Transcript not generated"})

        with open(transcript_file, "r") as f:
            transcript = f.read()

        # Cleanup
        os.remove(tmp_filename)
        os.remove(transcript_file)

        return {"transcript": transcript.strip()}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
