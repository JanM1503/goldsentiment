from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from models.finbert_gold import SentimentScores

# How strongly to amplify the average sentiment (pos-neg) before normalizing.
# 1.0 = no amplification; values in ~[1.5, 2.5] make the index move more.
SENSITIVITY = 2.2


@dataclass
class IndexComponents:
    nw: float
    nw_norm: float
    gsi: float
    classification: str


def _avg_scores(
    scores: List[SentimentScores],
    weights: List[float] | None = None,
) -> Tuple[float, float, float]:
    """Average or weighted-average sentiment scores.

    If ``weights`` is provided, uses a weighted mean and ignores items with
    non-positive weights.
    """

    if not scores:
        return 0.0, 0.0, 1.0

    if weights is None:
        n = float(len(scores))
        pos = sum(s.positive for s in scores) / n
        neg = sum(s.negative for s in scores) / n
        neu = sum(s.neutral for s in scores) / n
        return pos, neg, neu

    assert len(scores) == len(weights)
    total_w = sum(w for w in weights if w > 0)
    if total_w <= 0:
        return 0.0, 0.0, 1.0

    pos = sum(s.positive * w for s, w in zip(scores, weights) if w > 0) / total_w
    neg = sum(s.negative * w for s, w in zip(scores, weights) if w > 0) / total_w
    neu = sum(s.neutral * w for s, w in zip(scores, weights) if w > 0) / total_w
    return pos, neg, neu


def _classify_gsi(gsi: float) -> str:
    """Map GSI value into sentiment regime buckets.

    0–25   = Extremely Bearish
    25–45  = Bearish
    45–55  = Neutral
    55–75  = Bullish
    75–100 = Extremely Bullish
    """

    if gsi < 25:
        return "Extremely Bearish"
    if gsi < 45:
        return "Bearish"
    if gsi < 55:
        return "Neutral"
    if gsi < 75:
        return "Bullish"
    return "Extremely Bullish"


def compute_index(
    news_scores: List[SentimentScores],
    news_weights: List[float] | None = None,
) -> IndexComponents:
    """Compute Gold Sentiment Index from per-document SentimentScores (news-only).

    NW = avg(pos - neg) over news docs, in [-1, 1]

    NW_norm = (NW + 1) * 50
    """

    nw_pos, nw_neg, _ = _avg_scores(news_scores, news_weights)

    # Base sentiment in [-1, 1] as pos - neg.
    nw_raw = float(nw_pos - nw_neg)

    # Amplify extremes a bit so the index reacts more when the model is
    # confident, while clipping to [-1, 1].
    nw = max(-1.0, min(1.0, SENSITIVITY * nw_raw))

    # Normalize from [-1, 1] -> [0, 100]
    nw_norm = (nw + 1.0) * 50.0

    gsi = nw_norm
    classification = _classify_gsi(gsi)

    return IndexComponents(
        nw=nw,
        nw_norm=nw_norm,
        gsi=gsi,
        classification=classification,
    )
