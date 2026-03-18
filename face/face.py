# Wav2lip for generating talking head video
# face.py
import subprocess
import os
import gdown
from config import AVATAR_SILENT_VIDEO
from logger import logger, log_performance


WAV2LIP_DIR = os.path.join(os.path.dirname(__file__), "wav2lip")
WAV2LIP_CHECKPOINT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "wav2lip_gan.pth")


class LipSyncGenerator:
    def __init__(self, checkpoint_path=WAV2LIP_CHECKPOINT):
        self.checkpoint = os.path.abspath(checkpoint_path)
        if not os.path.exists(self.checkpoint):
            logger.info("Downloading Wav2Lip checkpoint...")
            gdown.download(
                "https://drive.google.com/uc?id=1P4ifX9RE1HAbAXVZeNw-NbcS1Vahkmtz",
                self.checkpoint,
                quiet=False,
            )
            logger.info("Checkpoint downloaded")

    @log_performance
    def generate(self, face_video_path, audio_path, output_path, pads=[0, 0, 0, 0]):
        """
        Generate lip-synced video.
        """
        logger.info(f"Generating lip-sync video: {output_path}")
        inference_script = os.path.join(WAV2LIP_DIR, "inference.py")
        cmd = [
            "python",
            inference_script,
            "--checkpoint_path",
            self.checkpoint,
            "--face",
            face_video_path,
            "--audio",
            audio_path,
            "--outfile",
            output_path,
            "--pads",
            str(pads[0]),
            str(pads[1]),
            str(pads[2]),
            str(pads[3]),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=WAV2LIP_DIR)
            logger.info(f"Lip-sync video generated: {output_path}")
            return output_path
        except subprocess.CalledProcessError as e:
            logger.error(f"Wav2Lip failed: {e.stderr}")
            raise
