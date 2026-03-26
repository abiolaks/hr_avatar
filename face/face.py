# face/face.py
#
# Loads Wav2Lip GAN + RetinaFace once at startup (in __init__).
# Each generate() call injects an args namespace and calls inference.main()
# directly — no subprocess spawn, no model reload per request.
#
# Before: subprocess + model reload = ~5-7s wasted per request
# After:  in-process call with cached models = ~0s overhead

import os
import sys
import types
import torch

from logger import logger, log_performance

WAV2LIP_DIR        = os.path.join(os.path.dirname(__file__), "wav2lip")
WAV2LIP_CHECKPOINT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "wav2lip_gan.pth")


class LipSyncGenerator:
    def __init__(self, checkpoint_path=WAV2LIP_CHECKPOINT):
        self.checkpoint = os.path.abspath(checkpoint_path)
        self._inference = None
        self._load_models()

    def _load_models(self):
        """Import inference.py once and load both models into memory."""
        if not os.path.exists(self.checkpoint):
            raise FileNotFoundError(
                f"Wav2Lip checkpoint not found: {self.checkpoint}\n"
                "Download it with:\n"
                "  curl -L 'https://huggingface.co/Nekochu/Wav2Lip/resolve/main/"
                "wav2lip_gan.pth?download=true' -o wav2lip_gan.pth"
            )

        # Make wav2lip importable
        if WAV2LIP_DIR not in sys.path:
            sys.path.insert(0, WAV2LIP_DIR)

        # inference.py writes intermediate files to temp/ relative to cwd
        os.makedirs(os.path.join(WAV2LIP_DIR, "temp"), exist_ok=True)

        orig_dir = os.getcwd()
        os.chdir(WAV2LIP_DIR)
        try:
            import importlib
            import inference as _inf
            importlib.reload(_inf)           # ensures clean globals on first load
            _inf.do_load(self.checkpoint)    # loads Wav2Lip GAN + RetinaFace into _inf globals
            self._inference = _inf
        finally:
            os.chdir(orig_dir)

        logger.info("Wav2Lip + RetinaFace models loaded and cached in memory")

    @log_performance
    def generate(self, face_video_path, audio_path, output_path, pads=[0, 0, 0, 0]):
        """
        Run lip-sync inference using pre-loaded in-memory models.
        No subprocess. No model reload. ~5-7s faster per request vs old version.
        """
        logger.info(f"Generating lip-sync video: {output_path}")

        is_gpu = torch.cuda.is_available()

        # inference.main() reads these as module-level globals — inject before each call
        self._inference.args = types.SimpleNamespace(
            checkpoint_path    = self.checkpoint,
            face               = os.path.abspath(face_video_path),
            audio              = os.path.abspath(audio_path),
            outfile            = os.path.abspath(output_path),
            static             = False,
            fps                = 25.0,
            pads               = pads,
            wav2lip_batch_size = 128 if is_gpu else 32,
            resize_factor      = 1,
            out_height         = 480,
            crop               = [0, -1, 0, -1],
            box                = [-1, -1, -1, -1],
            rotate             = False,
            nosmooth           = True,   # skip temporal smoothing — faster, no visible difference
            img_size           = 96,
        )

        orig_dir = os.getcwd()
        os.chdir(WAV2LIP_DIR)    # inference.py resolves temp/ relative to cwd
        try:
            self._inference.main()
        finally:
            os.chdir(orig_dir)

        logger.info(f"Lip-sync video generated: {output_path}")
        return output_path
