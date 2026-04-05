from fastapi import FastAPI
from pydantic import BaseModel
import base64
import numpy as np
import librosa
import io

app = FastAPI()

class AudioRequest(BaseModel):
    audio_id: str
    audio_base64: str


def compute_stats(data):
    flat = data.flatten()

    return {
        "rows": int(data.shape[0]),
        "columns": int(data.shape[1]) if len(data.shape) > 1 else 1,
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "variance": float(np.var(flat)),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "median": float(np.median(flat)),
        "mode": float(np.bincount(flat.astype(int)).argmax()) if len(flat) > 0 else 0,
        "range": float(np.max(flat) - np.min(flat)),
        "allowed_values": list(np.unique(flat)[:50]),  # limit size
        "value_range": [float(np.min(flat)), float(np.max(flat))],
        "correlation": float(np.corrcoef(flat[:1000], flat[:1000])[0, 1]) if len(flat) > 1 else 1.0
    }


@app.post("/")
async def process_audio(req: AudioRequest):
    # Decode base64
    audio_bytes = base64.b64decode(req.audio_base64)

    # Load audio
    audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)

    # Convert to dataset (example: MFCC features)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)

    stats = compute_stats(mfcc)

    return stats
