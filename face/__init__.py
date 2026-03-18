import sys
import os

_wav2lip_path = os.path.join(os.path.dirname(__file__), "wav2lip")
if os.path.isdir(_wav2lip_path) and _wav2lip_path not in sys.path:
    sys.path.insert(0, _wav2lip_path)

from .face import LipSyncGenerator
