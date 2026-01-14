from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone, timedelta, date
from pathlib import Path
from typing import Any, Dict, List

from models.finbert_gold import SentimentScores, analyze_batch
from processing.index_calc import IndexComponents, compute_index


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NEWS_JSON_PATH = PROJECT_ROOT / "news.json" 
SENTIMENT_RESULTS_PATH = PROJECT_ROOT / "docs" / "sentiment_results.json"
GSI_VALUE_PATH = PROJECT_ROOT / "docs" / "gsi_value.json"

HIGH_IMPACT_KEYWORDS = [
    "powell",
    "federal reserve",
    "fed",
    "rate hike",
    "rate cut",
    "rate hikes",
    "rate cuts",
    "interest rates",
    "monetary policy",
    "qe",
    "quantitative easing",
    "taper",
    "central bank",
    "central banks",
    "inflation shock",
    "stagflation",
    "recession",
    "crisis",
    "credit crunch",
    "de-dollarization",
    "brics",
]


@dataclass
class DocumentSentiment:
    source: str  # "news"
    id: str
    timestamp: str
    text: str
    sentiment: SentimentScores

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "source": self.source,
            "id": self.id,
            "timestamp": self.timestamp,
            "text": self.text,
        }
        d.update({
            "positive": self.sentiment.positive,
            "negative": self.sentiment.negative,
            "neutral": self.sentiment.neutral,
        })
        return d


def _load_json(path: Path) -> Any:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _load_news() -> List[Dict[str, Any]]:
    return list(_load_json(NEWS_JSON_PATH) or [])


def _extract_news_text(article: Dict[str, Any]) -> str:
    parts = [
        article.get("title") or "",
        article.get("description") or "",
        article.get("content") or "",
    ]
    return " \n".join(p for p in parts if p)


def _recency_weight(ts_str: str) -> float:
    """Return a recency weight in [0, 1] based on how old the item is.

    Heuristic rules (days old → weight):
      0–1   → 1.0   (very fresh)
      1–3   → 0.8
      3–7   → 0.6
      7–14  → 0.3
      14–30 → 0.1
      >30   → 0.0  (ignored for current sentiment)
    """

    if not ts_str:
        return 0.0

    try:
        ts = datetime.fromisoformat(str(ts_str).replace("Z", "+00:00"))
    except Exception:
        return 0.0

    now = datetime.now(timezone.utc)
    age_days = max(0.0, (now - ts).total_seconds() / 86400.0)

    if age_days <= 1:
        return 1.0
    if age_days <= 3:
        return 0.8
    if age_days <= 7:
        return 0.6
    if age_days <= 14:
        return 0.3
    if age_days <= 30:
        return 0.1
    return 0.0


def _impact_weight(score: SentimentScores, text: str) -> float:
    """Return an impact weight for a document.

    Combines:
      - confidence: |positive - negative| in [0, 1]
      - impact keywords: Fed/Powell/rates/crisis/etc. get a boost
      - non-linear emphasis for big moves (power > 1)
    """

    margin = abs(float(score.positive) - float(score.negative))
    # Base confidence in [0, 1]
    base = max(margin, 1e-3)

    txt = text.lower()
    impact_boost = 1.0
    if any(k in txt for k in HIGH_IMPACT_KEYWORDS):
        impact_boost = 3.0  # macro headline like Powell moves the needle more

    # Non-linear emphasis: small margins shrink, large margins grow.
    gamma = 1.5
    return (base ** gamma) * impact_boost


def analyze_documents() -> Dict[str, Any]:
    """Run FinBERT-Gold over recent news, compute index, and return results.

    This uses all articles in ``news.json`` with a positive recency weight,
    then applies impact weighting so that strong, macro-relevant headlines
    move the index more.
    """

    news_raw = _load_news()

    news_items: List[Dict[str, Any]] = []
    recency_weights: List[float] = []
    texts: List[str] = []
    for n in news_raw:
        w = _recency_weight(n.get("timestamp", ""))
        if w <= 0:
            continue
        text = _extract_news_text(n)
        if not text.strip():
            continue
        news_items.append(n)
        recency_weights.append(w)
        texts.append(text)

    news_scores: List[SentimentScores] = analyze_batch(texts) if texts else []

    # Combine recency and impact (confidence + macro keywords) into a single
    # effective weight per doc.
    effective_weights: List[float] = []
    for s, w, text in zip(news_scores, recency_weights, texts):
        impact = _impact_weight(s, text)
        effective_weights.append(w * impact)

    news_docs: List[DocumentSentiment] = []
    for raw, s in zip(news_items, news_scores):
        news_docs.append(
            DocumentSentiment(
                source="news",
                id=str(raw.get("url", "")),
                timestamp=str(raw.get("timestamp", "")),
                text=_extract_news_text(raw),
                sentiment=s,
            )
        )

    components: IndexComponents = compute_index(
        news_scores,
        news_weights=effective_weights,
    )

    now_iso = datetime.now(timezone.utc).isoformat()

    result = {
        "timestamp": now_iso,
        "news": {
            "count": len(news_docs),
            "documents": [d.to_dict() for d in news_docs],
            "nw": components.nw,
            "nw_norm": components.nw_norm,
        },
        "gsi": components.gsi,
        "classification": components.classification,
    }

    return result


def save_results(result: Dict[str, Any]) -> None:
    SENTIMENT_RESULTS_PATH.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    gsi_payload = {
        "timestamp": result["timestamp"],
        "gsi": result["gsi"],
        "classification": result["classification"],
        "nw_norm": result["news"]["nw_norm"],
    }
    GSI_VALUE_PATH.write_text(
        json.dumps(gsi_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def run_analysis_only() -> None:
    """Run analysis on existing tweets.json and news.json."""

    result = analyze_documents()
    save_results(result)


def run_full_pipeline() -> None:
    """Placeholder kept for symmetry; scraping is orchestrated in run.py."""

    run_analysis_only()


