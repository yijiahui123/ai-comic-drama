"""SadTalker API server.

Wraps the SadTalker Gradio engine in a FastAPI endpoint for lip-sync.

Usage:
    conda activate sadtalker
    python /Users/jeff/backend/ai-comic-drama/services/sadtalker_server.py
"""
import base64
import os
import sys
import tempfile

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

# Add SadTalker to path
SADTALKER_ROOT = os.path.expanduser("~/backend/SadTalker")
sys.path.insert(0, SADTALKER_ROOT)

# Shim for torchvision >= 0.26 which removed functional_tensor
import types
import torchvision.transforms.functional as _F
_mod = types.ModuleType("torchvision.transforms.functional_tensor")
_mod.rgb_to_grayscale = _F.rgb_to_grayscale
sys.modules["torchvision.transforms.functional_tensor"] = _mod

from src.gradio_demo import SadTalker

app = FastAPI(title="SadTalker API")

# Lazy-load engine
_sadtalker = None


def get_sadtalker() -> SadTalker:
    global _sadtalker
    if _sadtalker is None:
        _sadtalker = SadTalker(
            checkpoint_path=os.path.join(SADTALKER_ROOT, "checkpoints"),
            config_path=os.path.join(SADTALKER_ROOT, "src", "config"),
        )
    return _sadtalker


class LipsyncRequest(BaseModel):
    video_base64: str
    audio_base64: str
    shot_id: str = "unknown"


@app.post("/api/lipsync")
async def lipsync(req: LipsyncRequest):
    st = get_sadtalker()
    work_dir = tempfile.mkdtemp(prefix="sadtalker_")

    # Decode inputs to temp files
    video_path = os.path.join(work_dir, "input.mp4")
    audio_path = os.path.join(work_dir, "input.wav")
    with open(video_path, "wb") as f:
        f.write(base64.b64decode(req.video_base64))
    with open(audio_path, "wb") as f:
        f.write(base64.b64decode(req.audio_base64))

    # Extract first frame as source image
    import cv2

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return {"error": "Could not read video frame"}
    source_path = os.path.join(work_dir, "source.png")
    cv2.imwrite(source_path, frame)

    result_dir = os.path.join(work_dir, "output")
    os.makedirs(result_dir, exist_ok=True)

    # Run SadTalker
    try:
        result_path = st.test(
            source_image=source_path,
            driven_audio=audio_path,
            preprocess="full",
            still_mode=True,
            use_enhancer=False,
            batch_size=1,
            size=256,
            result_dir=result_dir,
        )
    except Exception as e:
        return {"error": f"SadTalker inference failed: {e}"}

    if result_path and os.path.exists(result_path):
        with open(result_path, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode()
        return {"video_base64": video_b64}

    return {"error": "Lip-sync generation failed"}


@app.get("/")
async def health():
    return {"status": "ok", "service": "SadTalker"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
