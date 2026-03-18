# vad.py
import torch
import pyaudio
import numpy as np
import wave
import threading
from queue import Queue
from logger import logger, log_performance


class VADetector:
    def __init__(self, sample_rate=16000, chunk=512, speech_threshold=0.5):
        self.sample_rate = sample_rate
        self.chunk = chunk
        self.speech_threshold = speech_threshold
        self.audio_queue = Queue()
        self.is_running = False
        self.thread = None

        # Load Silero VAD
        logger.info("Loading Silero VAD model...")
        self.model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=False
        )
        (self.get_speech_timestamps, _, _, _, _) = utils
        logger.info("VAD model loaded")

        # PyAudio
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sample_rate,
            input=True,
            frames_per_buffer=chunk,
        )

    def _record_loop(self):
        """Continuously read audio and detect speech."""
        speech_buffer = []
        silence_frames = 0
        is_speaking = False
        SILENCE_LIMIT = 20  # ~640ms

        logger.info("VAD recording loop started")
        while self.is_running:
            try:
                audio_bytes = self.stream.read(self.chunk, exception_on_overflow=False)
            except Exception as e:
                logger.error(f"Error reading audio: {e}")
                continue

            audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_float32 = audio_int16.astype(np.float32) / 32768.0

            speech_prob = self.model(
                torch.from_numpy(audio_float32), self.sample_rate
            ).item()

            if speech_prob > self.speech_threshold:
                if not is_speaking:
                    is_speaking = True
                    speech_buffer = [audio_int16]
                    silence_frames = 0
                else:
                    speech_buffer.append(audio_int16)
            else:
                if is_speaking:
                    silence_frames += 1
                    if silence_frames > SILENCE_LIMIT:
                        full_audio = np.concatenate(speech_buffer)
                        self.audio_queue.put(full_audio)
                        logger.debug(
                            f"Speech segment queued, length {len(full_audio)} samples"
                        )
                        is_speaking = False
                        speech_buffer = []
                        silence_frames = 0

    def start(self):
        self.is_running = True
        self.thread = threading.Thread(target=self._record_loop)
        self.thread.start()
        logger.info("VAD started")

    def stop(self):
        self.is_running = False
        if self.thread:
            self.thread.join()
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        logger.info("VAD stopped")

    @log_performance
    def get_speech_segment(self, timeout=5):
        """Retrieve next speech segment (blocking). Returns numpy array or None."""
        try:
            audio = self.audio_queue.get(timeout=timeout)
            return audio
        except:
            logger.debug("No speech segment received within timeout")
            return None

    def save_speech(self, audio, filename):
        """Save numpy array to WAV file."""
        wf = wave.open(filename, "wb")
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(self.sample_rate)
        wf.writeframes(audio.tobytes())
        wf.close()
        logger.info(f"Speech saved to {filename}")
