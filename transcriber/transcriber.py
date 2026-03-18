# transcriber.py
from faster_whisper import WhisperModel
from logger import logger, log_performance

class Transcriber:
    def __init__(self, model_size="base"):
        logger.info(f"Loading Whisper model '{model_size}' on CPU...")
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
        logger.info("Whisper model loaded")

    @log_performance
    def transcribe(self, audio_path):
        """Transcribe audio file and return text."""
        logger.info(f"Transcribing {audio_path}")
        segments, info = self.model.transcribe(audio_path, beam_size=5)
        text = " ".join([seg.text for seg in segments])
        logger.info(f"Transcription result: {text[:100]}...")
        return text
