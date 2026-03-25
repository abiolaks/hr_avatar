# Orchestrates the conversation flow with performance logging
# main.py
import uuid
import time
import os
from config import ASSETS_DIR, AVATAR_SILENT_VIDEO
from logger import logger, log_performance
from vad.vad import VADetector
from transcriber.transcriber import Transcriber
from brain.agent import HRAgent
from voice.voice import VoiceSynthesizer
from face.face import LipSyncGenerator


class HRAvatar:
    def __init__(self):
        logger.info("Initializing HRAvatar...")
        self.vad = VADetector()
        self.transcriber = Transcriber()
        self.agent = HRAgent()
        self.voice = VoiceSynthesizer()
        self.lipsync = LipSyncGenerator()
        self.silent_video = AVATAR_SILENT_VIDEO
        if not os.path.exists(self.silent_video):
            logger.error(f"Silent video not found: {self.silent_video}")
            raise FileNotFoundError(f"Missing silent video: {self.silent_video}")
        logger.info("HRAvatar initialized")

    @log_performance
    def process_user_input(self):
        """Listen, transcribe, get answer, synthesize, lip-sync, return video path."""
        logger.info("Waiting for speech...")
        audio = self.vad.get_speech_segment(timeout=10)
        if audio is None:
            logger.info("No speech detected")
            return None

        # Save temp audio
        temp_audio = f"/tmp/user_{uuid.uuid4()}.wav"
        self.vad.save_speech(audio, temp_audio)

        # Transcribe
        text = self.transcriber.transcribe(temp_audio)
        logger.info(f"User said: {text}")

        # Brain
        answer = self.agent.run(text)
        logger.info(f"Agent answer: {answer}")

        # Synthesize speech
        temp_voice = f"/tmp/voice_{uuid.uuid4()}.wav"
        self.voice.synthesize(answer, output_path=temp_voice)

        # Lip-sync video
        temp_video = f"/tmp/output_{uuid.uuid4()}.mp4"
        self.lipsync.generate(self.silent_video, temp_voice, temp_video)

        # Clean up temp audio/voice files (optional)
        os.unlink(temp_audio)
        os.unlink(temp_voice)

        return temp_video

    def run_conversation(self):
        """Main loop: process one user input at a time, play video."""
        self.vad.start()
        try:
            while True:
                video_path = self.process_user_input()
                if video_path:
                    logger.info(f"Playing video: {video_path}")
                    os.system(f"open {video_path}")  # macOS
                    # Optional: wait for video to finish before next iteration
                    time.sleep(2)
                else:
                    logger.info("No speech, exiting loop.")
                    break
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.vad.stop()
            self.agent.reset_conversation()
            logger.info("Conversation ended")


if __name__ == "__main__":
    avatar = HRAvatar()

    from brain.rag import RAGManager
    rag = RAGManager()

    # Ingest from local hr_docs/ AND Azure Blob Storage (if configured).
    # Set AZURE_STORAGE_CONNECTION_STRING + AZURE_STORAGE_CONTAINER env vars
    # to enable Azure. Local-only works without any env vars.
    rag.ingest_all(local_path="./hr_docs")

    avatar.run_conversation()
