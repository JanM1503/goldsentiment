from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# NOTE: The gold LoRA adapter currently produces nearly constant probabilities
# on this environment, likely due to classifier shape mismatches. For a
# reliable, high-variance signal we fall back to the base FinBERT-tone model,
# which is widely used for financial sentiment.
MODEL_NAME = "yiyanghkust/finbert-tone"


@dataclass
class SentimentScores:
    positive: float
    negative: float
    neutral: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "positive": float(self.positive),
            "negative": float(self.negative),
            "neutral": float(self.neutral),
        }


_tokenizer: Optional[AutoTokenizer] = None
_model: Optional[AutoModelForSequenceClassification] = None
_device: Optional[torch.device] = None


def _ensure_model_loaded() -> None:
    global _tokenizer, _model, _device
    if _tokenizer is not None and _model is not None and _device is not None:
        return

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    _model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    _model.to(_device)
    _model.eval()


def _softmax(logits: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.softmax(logits, dim=-1)


def _extract_scores(id2label: Mapping[int, str], probs: Sequence[float]) -> SentimentScores:
    # Map labels to lower-case for robustness (e.g., "Positive" vs "positive").
    label_map = {int(i): str(lbl).lower() for i, lbl in id2label.items()}

    pos = neg = neu = 0.0
    for idx, p in enumerate(probs):
        label = label_map.get(idx, "")
        if label == "positive":
            pos += float(p)
        elif label == "negative":
            neg += float(p)
        elif label == "neutral":
            neu += float(p)
        # Any "none" / other class is ignored for index purposes.

    # Ensure numeric types
    return SentimentScores(positive=pos, negative=neg, neutral=neu)


def analyze_text(text: str) -> SentimentScores:
    """Run FinBERT Gold on a single text and return sentiment scores.

    Returns scores as probabilities for positive/negative/neutral in [0, 1].
    """

    if not text:
        return SentimentScores(positive=0.0, negative=0.0, neutral=1.0)

    _ensure_model_loaded()
    assert _tokenizer is not None and _model is not None and _device is not None

    inputs = _tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding=True,
    ).to(_device)

    with torch.no_grad():
        outputs = _model(**inputs)
        logits = outputs.logits
        probs = _softmax(logits)[0].detach().cpu().tolist()

    id2label = getattr(_model.config, "id2label", {i: str(i) for i in range(len(probs))})
    return _extract_scores(id2label, probs)


def analyze_batch(texts: List[str]) -> List[SentimentScores]:
    """Analyze a batch of texts and return a list of SentimentScores."""

    results: List[SentimentScores] = []
    for t in texts:
        results.append(analyze_text(t))
    return results


__all__ = ["SentimentScores", "analyze_text", "analyze_batch"]
