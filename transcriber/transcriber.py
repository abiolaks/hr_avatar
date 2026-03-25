# transcriber.py
from faster_whisper import WhisperModel
from logger import logger, log_performance

class Transcriber:
    def __init__(self, model_size="base"):
        import torch
        if torch.cuda.is_available():
            device, compute_type = "cuda", "float16"
        else:
            device, compute_type = "cpu", "int8"
        logger.info(f"Loading Whisper model '{model_size}' on {device}...")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        logger.info("Whisper model loaded")

    @log_performance
    def transcribe(self, audio_path):
        """Transcribe audio file and return text."""
        logger.info(f"Transcribing {audio_path}")
        segments, info = self.model.transcribe(audio_path, beam_size=5)
        text = " ".join([seg.text for seg in segments])
        logger.info(f"Transcription result: {text[:100]}...")
        return text
