# voice.py
import torch
from TTS.api import TTS
import numpy as np
import soundfile as sf
from config import VOICE_SAMPLE
from logger import logger, log_performance

class VoiceSynthesizer:
    def __init__(self, speaker_wav_path=VOICE_SAMPLE, model_name="tts_models/multilingual/multi-dataset/xtts_v2"):
        # MPS does not support aten::_fft_r2c needed by XTTS — use CPU
        self.device = "cpu"
        logger.info(f"Loading XTTS on {self.device}...")
        self.tts = TTS(model_name).to(self.device)
        self.speaker_wav = speaker_wav_path
        logger.info("XTTS loaded")

    @log_performance
    def synthesize(self, text, output_path=None):
        """
        Generate speech from text. If output_path is None, returns numpy array.
        """
        wav = self.tts.tts(text=text, speaker_wav=self.speaker_wav, language="en")
        if output_path:
            sf.write(output_path, wav, samplerate=24000)
            logger.info(f"Synthesized audio saved to {output_path}")
            return output_path
        return np.array(wav)

    def stream_chunks(self, text, chunk_size=100):
        """Generator for streaming (simulated)."""
        yield self.synthesize(text)
