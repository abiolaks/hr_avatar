# Fastapi app for the browser interaction
# web/app.py
import sys
sys.path.append("..")
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
import uvicorn
import uuid
import os
from integration.main import HRAvatar  # reuse the avatar class

app = FastAPI()
avatar = HRAvatar()  # loads all modules (may take time)

@app.get("/", response_class=HTMLResponse)
async def index():
    return """
    <html>
        <body>
            <h2>HR Avatar</h2>
            <form action="/ask" method="post" enctype="multipart/form-data">
                <input type="file" name="audio" accept="audio/wav" required>
                <button type="submit">Ask</button>
            </form>
        </body>
    </html>
    """

@app.post("/ask")
async def ask(audio: UploadFile = File(...)):
    # Save uploaded audio
    temp_audio = f"/tmp/user_{uuid.uuid4()}.wav"
    with open(temp_audio, "wb") as f:
        f.write(await audio.read())

    # Use avatar's pipeline but we need direct access to methods
    # For simplicity, replicate the steps here (or refactor)
    text = avatar.transcriber.transcribe(temp_audio)
    answer = avatar.agent.run(text)
    temp_voice = f"/tmp/voice_{uuid.uuid4()}.wav"
    avatar.voice.synthesize(answer, output_path=temp_voice)
    temp_video = f"/tmp/output_{uuid.uuid4()}.mp4"
    avatar.lipsync.generate(avatar.silent_video, temp_voice, temp_video)

    return FileResponse(temp_video, media_type="video/mp4")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
