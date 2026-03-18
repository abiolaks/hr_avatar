# Blockers & Solutions

A running log of issues encountered during setup and development, with their fixes.

---

## 1. `wav2lip` — Repository Not Found

**Error:**
```
remote: Repository not found.
fatal: repository 'https://github.com/numz/wav2lip.git/' not found
```

**Cause:** The `numz/wav2lip` GitHub repository was deleted/made private.

**Fix:** `wav2lip` is not a pip-installable package (no `setup.py` / `pyproject.toml`). Clone it manually:
```bash
git clone https://github.com/justinjohn0306/Wav2Lip.git face/wav2lip
```
Remove it from `requirements.txt` entirely. `face/__init__.py` adds it to `sys.path` automatically.

---

## 2. `langgraph` — Conflicting `langchain-core` Versions

**Error:**
```
langchain 0.2.0 depends on langchain-core>=0.2.0
langgraph 0.0.20 depends on langchain-core<0.2.0
```

**Cause:** `langgraph 0.0.20` was designed for an older `langchain-core` incompatible with `langchain 0.2.0`.

**Fix:** Upgrade `langgraph` to `0.1.19`, which targets `langchain-core>=0.2.0`.

---

## 3. `av` — Cython Compile Error

**Error:**
```
Cannot assign type '...' to '...'. Exception values are incompatible.
Suggest adding 'noexcept' to the type of 'log_context_name'.
```

**Cause:** `faster-whisper 0.9.0` pulls in `av==10.0.0`, which fails to compile against newer Cython.

**Fix:** Upgrade `faster-whisper` to `1.0.3`, which depends on `av>=12.0.0` where the Cython issue is resolved.

---

## 4. `pkg-config` Missing — PyAV Build Failure

**Error:**
```
pkg-config is required for building PyAV
```

**Cause:** `pkg-config` and `ffmpeg` were not installed on the system.

**Fix:**
```bash
brew install pkg-config ffmpeg
```

---

## 5. `TTS` — Exact `numpy` Pin Conflicts with `chromadb`

**Error:**
```
tts 0.22.0 depends on numpy==1.22.0; python_version <= "3.10"
chromadb 0.4.22 depends on numpy>=1.22.5
```

**Cause:** `TTS 0.22.0` hard-pins `numpy==1.22.0` exactly on Python ≤ 3.10. No single numpy version satisfies both.

**Fix:** Remove `TTS` from `requirements.txt`. Install separately bypassing its numpy pin:
```bash
pip install TTS==0.22.0 --no-deps
```
TTS works fine at runtime with numpy ≥ 1.22.5.

---

## 6. `torch` — No Matching Distribution + `facenet-pytorch` Conflict

**Error:**
```
facenet-pytorch 2.6.0 depends on torch>=2.2.0,<2.3.0
No matching distributions available for torch==2.0.1 (environment: macOS arm64)
```

**Cause:** `torch 2.0.1` has no wheel for macOS Apple Silicon, and `facenet-pytorch 2.6.0` requires `torch>=2.2.0`.

**Fix:** Upgrade the torch stack:
```
torch==2.2.2
torchvision==0.17.2
torchaudio==2.2.2
```

---

## 7. `logger.py` — Missing `os` Import and `Path` Division Error

**Error:**
```
NameError: name 'os' is not defined
TypeError: unsupported operand type(s) for /: 'str' and 'str'
```

**Cause:** `logger.py` used `os.path.join` without importing `os`, and used `LOGS_DIR / "file.log"` (Path division syntax) on a plain string.

**Fix:**
- Add `import os` at the top of `logger.py`
- Change `LOGS_DIR / "hr_avatar.log"` → `os.path.join(LOGS_DIR, "hr_avatar.log")`

---

## 8. `ModuleNotFoundError` — Package Imports Inside `__init__.py`

**Error:**
```
ImportError: cannot import name 'VADetector' from 'vad'
```

**Cause:** `vad/__init__.py` used `from vad.vad import VADetector` — absolute import inside a package causes a circular naming conflict.

**Fix:** Use relative imports in all `__init__.py` files:
```python
from .vad import VADetector       # vad/__init__.py
from .voice import VoiceSynthesizer  # voice/__init__.py
from .transcriber import Transcriber  # transcriber/__init__.py
from .face import LipSyncGenerator   # face/__init__.py
```

---

## 9. `pkg_resources` Not Found — `librosa` Import Failure

**Error:**
```
ModuleNotFoundError: No module named 'pkg_resources'
```

**Cause:** `librosa 0.10.0` uses `pkg_resources.resource_filename` which is broken in newer environments even when `setuptools` is installed.

**Fix:** Upgrade librosa:
```bash
pip install "librosa==0.10.2"
```

---

## 10. TTS Missing Runtime Dependencies (`--no-deps` side effect)

**Error (repeated):**
```
ModuleNotFoundError: No module named 'pysbd'
ModuleNotFoundError: No module named 'trainer'
ModuleNotFoundError: No module named 'spacy'
...
```

**Cause:** Installing `TTS==0.22.0 --no-deps` skips all its dependencies. They must be installed manually.

**Fix:** Install all TTS runtime deps explicitly:
```bash
pip install coqpit trainer "transformers==4.44.2" einops encodec unidecode \
    inflect num2words pysbd anyascii spacy batch-face pypdf \
    "pandas>=1.4,<2.0" bangla bnnumerizer bnunicodenormalizer \
    hangul_romanize jamo jieba nltk pypinyin "gruut[de,es,fr]==2.2.3" \
    umap-learn cython flask g2pkk
```

---

## 11. `transformers` — `BeamSearchScorer` Removed in 4.47+

**Error:**
```
ImportError: cannot import name 'BeamSearchScorer' from 'transformers'
```

**Cause:** `transformers 4.57.x` removed `BeamSearchScorer` from the public API. TTS 0.22.0 uses it internally via XTTS.

**Fix:** Pin transformers to the last version that still exposes it:
```bash
pip install "transformers==4.44.2"
```

---

## 12. `ChatOllama` — `bind_tools` Not Implemented

**Error:**
```
AttributeError: 'Ollama' object has no attribute 'bind_tools'
NotImplementedError (from ChatOllama in langchain-community)
```

**Cause:** `langchain_community.llms.Ollama` is a legacy class with no tool support. `ChatOllama` in `langchain-community==0.2.0` also doesn't implement `bind_tools`.

**Fix:** Use the dedicated `langchain-ollama` package:
```bash
pip install "ollama>=0.3.0" "langchain-ollama==0.1.3"
```
Update import in `brain/agent.py`:
```python
from langchain_ollama import ChatOllama
```

---

## 13. MPS Device — `aten::_fft_r2c` Not Implemented

**Error:**
```
NotImplementedError: The operator 'aten::_fft_r2c' is not currently implemented for the MPS device.
```

**Cause:** XTTS uses FFT operations during speaker embedding that are not supported on Apple MPS (Metal GPU) in PyTorch 2.2.x.

**Fix:** Force XTTS to run on CPU in `voice/voice.py`:
```python
self.device = "cpu"  # MPS does not support aten::_fft_r2c needed by XTTS
```

---

## 14. Wav2Lip — `No module named 'wav2lip'`

**Error:**
```
/usr/bin/python: Error while finding module specification for 'wav2lip.inference'
ModuleNotFoundError: No module named 'wav2lip'
```

**Cause:** `face/face.py` called `python -m wav2lip.inference` but Wav2Lip is not an installed package.

**Fix:** Call `inference.py` as a direct script path:
```python
inference_script = os.path.join(WAV2LIP_DIR, "inference.py")
cmd = ["python", inference_script, ...]
```

---

## 15. Wav2Lip — `checkpoints/mobilenet.pth` Not Found

**Error:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'checkpoints/mobilenet.pth'
```

**Cause:** Two issues combined:
1. `mobilenet.pth` (face detection model) was never downloaded
2. The subprocess ran from the project root so the relative path `checkpoints/` resolved incorrectly

**Fix:**
1. Download the model:
```bash
curl -L "https://huggingface.co/py-feat/retinaface/resolve/main/mobilenet0.25_Final.pth" \
    -o face/wav2lip/checkpoints/mobilenet.pth
```
2. Run subprocess from the wav2lip directory and use absolute path for checkpoint:
```python
subprocess.run(cmd, check=True, cwd=WAV2LIP_DIR)
self.checkpoint = os.path.abspath(checkpoint_path)
```

---

## 16. RAG Ingesting 0 Documents

**Symptom:** Logs showed `Loaded 0 documents` even though `hr_docs/` had a PDF file.

**Cause:** `brain/rag.py` only loaded `**/*.txt` files. The HR policy document is a `.pdf`.

**Fix:** Add PDF loading support:
```python
from langchain_community.document_loaders import PyPDFLoader
pdf_loader = DirectoryLoader(docs_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
documents += pdf_loader.load()
```
Also install: `pip install pypdf`

---

## 17. Wav2Lip Checkpoint — Google Drive / SharePoint Links Dead

**Error:**
```
gdown.exceptions.FileURLRetrievalError: Cannot retrieve the public link of the file.
404 FILE NOT FOUND (SharePoint link)
```

**Cause:** Both the original Google Drive ID in `face/face.py` and the SharePoint link from the Wav2Lip README were dead/expired.

**Fix:** Download from HuggingFace instead:
```bash
curl -L "https://huggingface.co/Nekochu/Wav2Lip/resolve/main/wav2lip_gan.pth?download=true" \
    -o wav2lip_gan.pth
```

---

## Final Working Install Sequence

```bash
# 1. System deps
brew install pkg-config ffmpeg

# 2. Python deps
pip install -r requirements.txt

# 3. TTS (bypass numpy pin)
pip install TTS==0.22.0 --no-deps
pip install coqpit trainer "transformers==4.44.2" einops encodec unidecode \
    inflect num2words pysbd anyascii spacy batch-face pypdf \
    "pandas>=1.4,<2.0" bangla bnnumerizer bnunicodenormalizer \
    hangul_romanize jamo jieba nltk pypinyin "gruut[de,es,fr]==2.2.3" \
    umap-learn cython flask g2pkk "ollama>=0.3.0" "langchain-ollama==0.1.3"

# 4. Clone Wav2Lip
git clone https://github.com/justinjohn0306/Wav2Lip.git face/wav2lip

# 5. Download model checkpoints
curl -L "https://huggingface.co/Nekochu/Wav2Lip/resolve/main/wav2lip_gan.pth?download=true" \
    -o wav2lip_gan.pth
curl -L "https://huggingface.co/py-feat/retinaface/resolve/main/mobilenet0.25_Final.pth" \
    -o face/wav2lip/checkpoints/mobilenet.pth

# 6. Pull Ollama models
ollama pull qwen3:4b
ollama pull nomic-embed-text
```
