# TriGPT – Dev Quickstart

## 1. Project layout

- `src/` – Python backend (FastAPI, RAG, vision models)
- `models/` – saved PyTorch + classifier checkpoints
- `data/` – local data (faces, emotions, etc.)
- `.env` – config (AWS region, S3 bucket, etc.)

## 2. Start backend after opening VS Code

```bash
cd /Users/nhathuoc/Desktop/TriGPT
source .venv/bin/activate       # activate venv
uvicorn src.api:app --reload    # start API
