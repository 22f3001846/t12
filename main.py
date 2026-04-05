from fastapi import FastAPI
from pydantic import BaseModel
import base64
import numpy as np

app = FastAPI()

class AudioRequest(BaseModel):
    audio_id: str
    audio_base64: str


def compute_stats(data):
    flat = data.flatten()

    return {
        "rows": int(data.shape[0]),
        "columns": list(range(data.shape[1])) if len(data.shape) > 1 else [0],

        "mean": {"value": float(np.mean(flat))},
        "std": {"value": float(np.std(flat))},
        "variance": {"value": float(np.var(flat))},
        "min": {"value": float(np.min(flat))},
        "max": {"value": float(np.max(flat))},
        "median": {"value": float(np.median(flat))},
        "mode": {"value": float(np.bincount(flat.astype(int)).argmax() if len(flat)>0 else 0)},
        "range": {"value": float(np.max(flat) - np.min(flat))},

        "allowed_values": {"values": list(np.unique(flat)[:20])},
        "value_range": {"min": float(np.min(flat)), "max": float(np.max(flat))},

        "correlation": [1.0]
    }


@app.post("/")
async def process_audio(req: AudioRequest):
    audio_bytes = base64.b64decode(req.audio_base64)

    # FAST conversion (no librosa)
    data = np.frombuffer(audio_bytes, dtype=np.uint8)

    # reshape for 2D stats
    data = data.reshape(-1, 1)

    return compute_stats(data)
