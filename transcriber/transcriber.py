# transcriber.py
import os
import subprocess
import tempfile
import torch
from faster_whisper import WhisperModel
from logger import logger, log_performance


def _to_wav(audio_path: str) -> tuple[str, bool]:
    """
    Convert any audio file to a 16 kHz mono WAV that Whisper handles reliably.
    Returns (wav_path, should_delete).  If the file is already a .wav we
    return it unchanged and should_delete=False.
    """
    if audio_path.lower().endswith(".wav"):
        return audio_path, False

    fd, wav_path = tempfile.mkstemp(suffix="_converted.wav")
    os.close(fd)
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-i", audio_path,
            "-ar", "16000",   # 16 kHz — Whisper's native rate
            "-ac", "1",       # mono
            "-f", "wav",
            wav_path,
        ],
        capture_output=True,
    )
    if result.returncode != 0:
        os.unlink(wav_path)
        raise RuntimeError(
            f"ffmpeg conversion failed: {result.stderr.decode(errors='replace')}"
        )
    return wav_path, True


class Transcriber:
    def __init__(self, model_size=None):
        # large-v3 on GPU for best accuracy.
        # The model is unloaded from GPU immediately after each transcription
        # so that TTS + Wav2Lip have the full ~3.8 GiB back during lipsync.
        # Whisper and lipsync are sequential — they never compete for VRAM.
        self.device       = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if self.device == "cuda" else "int8"
        self.model_size   = model_size or "large-v3"
        self.model        = None
        self._load()

    def _load(self):
        logger.info(f"Loading Whisper '{self.model_size}' on {self.device}...")
        self.model = WhisperModel(
            self.model_size, device=self.device, compute_type=self.compute_type
        )
        logger.info("Whisper model loaded")

    def _unload(self):
        """Delete the model and free GPU VRAM for TTS + Wav2Lip."""
        del self.model
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Whisper unloaded — GPU VRAM freed for lipsync")

    @log_performance
    def transcribe(self, audio_path):
        """
        Transcribe audio, then immediately unload the model from GPU so that
        TTS + Wav2Lip have the headroom they need.  The model reloads from the
        local HuggingFace cache before the next transcription.
        """
        if self.model is None:
            self._load()

        logger.info(f"Transcribing {audio_path}")
        wav_path, should_delete = _to_wav(audio_path)
        try:
            segments, _ = self.model.transcribe(
                wav_path,
                language="en",
                beam_size=5,
                temperature=0,
                condition_on_previous_text=False,
                vad_filter=False,
                initial_prompt=(
                    "HR assistant, company policy, learning management system, "
                    "courses, skills assessment, department, job role, employee "
                    "training, course recommendation, workplace, enroll, certificate. "
                    "Python, SQL, JavaScript, pandas, NumPy, scikit-learn, "
                    "machine learning, data science, data analysis, deep learning, "
                    "Excel, Power BI, Tableau, cloud, Azure, AWS."
                ),
            )
            # Filter out segments where Whisper itself is uncertain — these are
            # typically noise bursts or inaudible fragments that produce gibberish text.
            # avg_logprob < -1.0 means the model had very low confidence.
            # no_speech_prob > 0.6 means the segment is more likely silence than speech.
            reliable = [
                seg for seg in segments
                if seg.avg_logprob > -1.0 and seg.no_speech_prob < 0.6
            ]
            text = " ".join(seg.text for seg in reliable).strip()
        finally:
            if should_delete and os.path.exists(wav_path):
                os.unlink(wav_path)
            # Always unload after inference — frees ~3.8 GiB for lipsync
            self._unload()

        logger.info(f"Transcription result: {text[:100]}...")
        return text
