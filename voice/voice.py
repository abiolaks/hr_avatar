# voice.py
import re
import torch
import numpy as np
import soundfile as sf
from TTS.api import TTS
from config import VOICE_SAMPLE
from logger import logger, log_performance

# Emoji: covers supplementary planes + common symbol blocks
_EMOJI_RE = re.compile(
    "[\U00010000-\U0010ffff"
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\u2600-\u26FF\u2700-\u27BF]+",
    flags=re.UNICODE,
)

def _clean_for_tts(text: str) -> str:
    """Strip emojis, markdown, and exclamation marks for clean TTS input."""
    text = _EMOJI_RE.sub("", text)
    # Keep link labels, drop URLs; also strip any bare https:// URLs
    text = re.sub(r"\[([^\]]*)\]\([^\)]*\)", r"\1", text)
    text = re.sub(r"https?://\S+", "", text)
    # Strip markdown symbols (bold/italic/headers/bullets/code)
    text = re.sub(
        r"(\*{1,3}|_{1,3})|`{1,3}[^`]*`{1,3}|^#{1,6}\s*|^\s*[-*+]\s+|^\s*\d+\.\s+|^[-_*]{3,}\s*$",
        " ", text, flags=re.MULTILINE,
    )
    text = text.replace("!", ".")        # exclamation → period for natural cadence
    text = re.sub(r"\s{2,}", " ", text)  # collapse extra spaces
    return text.strip()


class VoiceSynthesizer:
    def __init__(self, speaker_wav_path=VOICE_SAMPLE, model_name="tts_models/multilingual/multi-dataset/xtts_v2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading XTTS on {self.device}...")
        self.tts = TTS(model_name).to(self.device)
        self.speaker_wav = speaker_wav_path

        # Pre-compute and cache speaker conditioning latents once at startup.
        # tts.tts() recomputes these from speaker_wav on every call — skipping
        # that step saves ~2-3s per synthesis by calling the model's inference()
        # layer directly with the cached tensors.
        logger.info("Pre-computing speaker conditioning latents...")
        tts_model = self.tts.synthesizer.tts_model
        self._gpt_cond_latent, self._speaker_embedding = tts_model.get_conditioning_latents(
            audio_path=[speaker_wav_path],
            gpt_cond_len=30,
            gpt_cond_chunk_len=4,
            max_ref_length=60,
        )
        logger.info("XTTS loaded with cached speaker latents")

    @log_performance
    def synthesize(self, text, output_path=None):
        """
        Generate speech from text using pre-cached speaker conditioning.
        Bypasses speaker_wav re-loading on every call (~2-3s savings per request).
        """
        text = _clean_for_tts(text)
        if not text:
            text = "."
        # Hard cap: truncate at the last sentence boundary within 600 chars.
        # Prevents runaway LLM replies from causing 40+ second TTS + VRAM OOM.
        if len(text) > 600:
            cut = text.rfind('.', 0, 600)
            text = text[:cut + 1] if cut != -1 else text[:600]

        tts_model = self.tts.synthesizer.tts_model
        out = tts_model.inference(
            text=text,
            language="en",
            gpt_cond_latent=self._gpt_cond_latent,
            speaker_embedding=self._speaker_embedding,
            temperature=0.7,
            enable_text_splitting=True,
        )
        wav = out["wav"]

        # Release cached GPU memory so Ollama has room for LLM inference next turn
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if output_path:
            sf.write(output_path, np.array(wav), samplerate=24000)
            logger.info(f"Synthesized audio saved to {output_path}")
            return output_path
        return np.array(wav)

    def stream_chunks(self, text, chunk_size=100):
        """Generator for streaming (simulated)."""
        yield self.synthesize(text)
