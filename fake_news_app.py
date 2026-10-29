
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import re
import threading
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(os.getenv("MODEL_DIR", APP_DIR / "saved_fake_news_model"))
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "384"))
MAX_INPUT_CHARS = 30000

app = FastAPI(title="Fake News Detection Demo", description="Human-reviewed text classification with a fine-tuned Hugging Face model.", version="1.0.0")
templates = Jinja2Templates(directory=str(APP_DIR))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_tokenizer = None
_model = None
_load_lock = threading.Lock()


class NewsInput(BaseModel):
    headline: str = Field(default="", max_length=2000)
    body_text: str = Field(default="", max_length=MAX_INPUT_CHARS)


def clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_model():
    """Load once on demand, so the page opens before training has finished."""
    global _tokenizer, _model
    if _model is not None and _tokenizer is not None:
        return _tokenizer, _model
    with _load_lock:
        if _model is not None and _tokenizer is not None:
            return _tokenizer, _model
        if not MODEL_DIR.is_dir():
            raise HTTPException(
                status_code=503,
                detail=f"Model files not found at {MODEL_DIR}. Run the notebook through its save-model cell first.",
            )
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR), local_files_only=True)
            model.to(device)
            model.eval()
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Could not load the saved model: {exc}") from exc
        _tokenizer, _model = tokenizer, model
    return _tokenizer, _model


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index_fake_news.html", context={})


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "model_directory_exists": MODEL_DIR.is_dir(),
        "device": str(device),
    }


@app.post("/predict/")
async def predict(news: NewsInput):
    if len(news.headline) + len(news.body_text) > MAX_INPUT_CHARS:
        raise HTTPException(status_code=413, detail=f"Combined input must be at most {MAX_INPUT_CHARS} characters.")
    text = clean_text(f"{news.headline}\n{news.body_text}")
    if not text:
        raise HTTPException(status_code=422, detail="Enter a headline or article body.")

    tokenizer, model = load_model()
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(device)
    with torch.inference_mode():
        probabilities = torch.softmax(model(**encoded).logits, dim=-1)[0].cpu().tolist()

    labels = getattr(model.config, "id2label", {}) or {}
    fake_id = next((int(i) for i, name in labels.items() if str(name).upper() == "FAKE"), 0)
    real_id = next((int(i) for i, name in labels.items() if str(name).upper() == "REAL"), 1)
    predicted_id = int(max(range(len(probabilities)), key=probabilities.__getitem__))
    label_name = str(labels.get(predicted_id, {0: "FAKE", 1: "REAL"}.get(predicted_id, predicted_id))).upper()
    return {
        "predicted_label": label_name,
        "p_fake": float(probabilities[fake_id]),
        "p_real": float(probabilities[real_id]),
        "notice": "This is a model prediction, not fact verification. Review the article and its sources.",
    }
