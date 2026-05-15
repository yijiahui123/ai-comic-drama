"""ChatTTS API server.

Exposes a POST /generate_audio endpoint compatible with the VideoGenerator skill.

Usage:
    conda activate ai-comic
    HF_ENDPOINT=https://hf-mirror.com python services/chattts_server.py
"""
import base64
import io
import os
import wave

import ChatTTS
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="ChatTTS API")

# Load model once at startup
chat = ChatTTS.Chat()
chat.load(compile=False, source="huggingface")

# Pre-defined voice presets (seed-based).
# To add new voices: generate with different seed_id, listen, then add here.
VOICE_PRESETS = {
    "male_1":    {"seed_id": 1},
    "male_2":    {"seed_id": 42},
    "female_1":  {"seed_id": 2},
    "female_2":  {"seed_id": 3},
    "female_3":  {"seed_id": 100},
}


class AudioRequest(BaseModel):
    text: str
    speaker_id: int = 0          # seed for random voice generation
    voice_preset: str = ""        # name from VOICE_PRESETS (overrides speaker_id)
    ref_audio_path: str = ""      # path to reference audio for voice cloning (overrides both)
    speed: float = 1.0
    temperature: float = 0.3


@app.get("/voices")
async def list_voices():
    """List available voice presets."""
    return {"voices": list(VOICE_PRESETS.keys())}


@app.post("/generate_audio")
async def generate_audio(req: AudioRequest):
    # Determine speaker embedding: ref_audio > voice_preset > speaker_id seed
    if req.ref_audio_path and os.path.isfile(req.ref_audio_path):
        spk_emb = chat.sample_audio_speaker(req.ref_audio_path)
    elif req.voice_preset and req.voice_preset in VOICE_PRESETS:
        seed = VOICE_PRESETS[req.voice_preset]["seed_id"]
        torch.manual_seed(seed)
        spk_emb = chat.sample_random_speaker()
    else:
        torch.manual_seed(req.speaker_id)
        spk_emb = chat.sample_random_speaker()

    params = ChatTTS.Chat.InferCodeParams(
        spk_emb=spk_emb,
        temperature=req.temperature,
        top_P=0.7,
        top_K=20,
    )
    refine_params = ChatTTS.Chat.RefineTextParams(
        prompt="[oral_2][laugh_0][break_6]",
    )

    wavs = chat.infer(
        [req.text],
        params_infer_code=params,
        params_refine_text=refine_params,
        use_decoder=True,
    )

    audio_np = wavs[0]
    if isinstance(audio_np, torch.Tensor):
        audio_np = audio_np.numpy()
    audio_np = audio_np.flatten()

    # Convert to 16-bit PCM WAV
    sample_rate = 24000
    audio_int16 = (audio_np * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())

    audio_b64 = base64.b64encode(buf.getvalue()).decode()
    return {"audio_base64": audio_b64}


@app.get("/")
async def health():
    return {"status": "ok", "service": "ChatTTS"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9966)
