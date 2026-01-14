from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone, timedelta, date
from pathlib import Path
from typing import Any, Dict, List

from models.finbert_gold import SentimentScores, analyze_batch
from processing.index_calc import IndexComponents, compute_index
from scraping.newsapi import _calculate_gold_relevance_score


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

# Geopolitical crisis keywords - these are BULLISH for gold (safe haven)
# When FinBERT sees these as negative, we should flip or reduce the bearish signal
GEOPOLITICAL_CRISIS_KEYWORDS = [
    "war", "conflict", "military action", "invasion", "sanctions",
    "iran turmoil", "iran protests", "iran crisis", "middle east crisis",
    "geopolitical", "tensions", "threatened", "threats", "trump threatens",
    "oil lane disruption", "trade war", "tariffs", "embargo",
    "greenland", "mineral wealth", "venezuela", "military", "attack"
]

# EXTREME geopolitical events - full inversion (negative becomes positive)
EXTREME_CRISIS_KEYWORDS = [
    "trump threatens war", "trump vows military", "iran war", "venezuela invasion",
    "world war", "nuclear", "missile", "military strikes", "armed conflict"
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


def _impact_weight(score: SentimentScores, text: str, relevance_score: float = 0.5) -> float:
    """Return an impact weight for a document.

    Combines:
      - confidence: |positive - negative| in [0, 1]
      - impact keywords: Fed/Powell/rates/crisis/etc. get a boost
      - gold relevance: direct gold articles get higher weight
      - non-linear emphasis for big moves (power > 1)
    """

    margin = abs(float(score.positive) - float(score.negative))
    # Base confidence in [0, 1]
    base = max(margin, 1e-3)

    txt = text.lower()
    impact_boost = 1.0
    
    # REDUCED from 3.0 to 1.8 to decrease volatility
    if any(k in txt for k in HIGH_IMPACT_KEYWORDS):
        impact_boost = 1.8
    
    # Gold-direct articles get additional boost
    if relevance_score >= 0.7:  # Direct gold mention
        impact_boost *= 1.3

    # REDUCED gamma from 1.5 to 1.2 to decrease volatility
    gamma = 1.2
    
    # Multiply by relevance score so irrelevant articles have minimal impact
    return (base ** gamma) * impact_boost * relevance_score


def analyze_documents() -> Dict[str, Any]:
    """Run FinBERT-Gold over recent news, compute index, and return results.

    This uses all articles in ``news.json`` with a positive recency weight,
    then applies impact weighting so that strong, macro-relevant headlines
    move the index more.
    """

    news_raw = _load_news()

    news_items: List[Dict[str, Any]] = []
    recency_weights: List[float] = []
    relevance_scores: List[float] = []
    texts: List[str] = []
    
    for n in news_raw:
        w = _recency_weight(n.get("timestamp", ""))
        if w <= 0:
            continue
        
        # Calculate gold relevance score
        relevance = _calculate_gold_relevance_score(n)
        if relevance < 0.3:  # Filter out low-relevance articles
            continue
            
        text = _extract_news_text(n)
        if not text.strip():
            continue
            
        news_items.append(n)
        recency_weights.append(w)
        relevance_scores.append(relevance)
        texts.append(text)

    news_scores: List[SentimentScores] = analyze_batch(texts) if texts else []

    # Apply geopolitical crisis adjustment to scores
    # Geopolitical crises are BULLISH for gold (safe haven) but FinBERT sees them as negative
    adjusted_scores: List[SentimentScores] = []
    for s, text in zip(news_scores, texts):
        txt_lower = text.lower()
        
        # Check for extreme crisis (war, Trump threatens, etc.)
        is_extreme_crisis = any(kw in txt_lower for kw in EXTREME_CRISIS_KEYWORDS)
        is_geopolitical_crisis = any(kw in txt_lower for kw in GEOPOLITICAL_CRISIS_KEYWORDS)
        
        if is_extreme_crisis and s.negative > 0.3:
            # EXTREME crisis: Trump threatens war, Iran war, etc.
            # This is EXTREMELY BULLISH for gold - FULL INVERSION
            # Convert 90% of negative sentiment to positive
            crisis_adjustment = 0.9
            new_pos = min(s.positive + (s.negative * crisis_adjustment), 0.95)
            new_neg = s.negative * (1 - crisis_adjustment)
            new_neu = max(1.0 - new_pos - new_neg, 0.0)
            adjusted_scores.append(SentimentScores(
                positive=new_pos,
                negative=new_neg,
                neutral=new_neu
            ))
        elif is_geopolitical_crisis and s.negative > 0.5:
            # Regular crisis: sanctions, tensions, geopolitical risk
            # This is BULLISH for gold - partial inversion
            # Convert 70% of negative sentiment to positive (increased from 60%)
            crisis_adjustment = 0.7
            new_pos = min(s.positive + (s.negative * crisis_adjustment), 0.85)
            new_neg = s.negative * (1 - crisis_adjustment)
            new_neu = max(1.0 - new_pos - new_neg, 0.0)
            adjusted_scores.append(SentimentScores(
                positive=new_pos,
                negative=new_neg,
                neutral=new_neu
            ))
        else:
            adjusted_scores.append(s)

    # Combine recency, impact, and relevance into a single effective weight per doc.
    effective_weights: List[float] = []
    for s, w, text, relevance in zip(adjusted_scores, recency_weights, texts, relevance_scores):
        impact = _impact_weight(s, text, relevance)
        effective_weights.append(w * impact)

    news_docs: List[DocumentSentiment] = []
    for raw, s in zip(news_items, adjusted_scores):
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
        adjusted_scores,
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
    
    # Generate and log feedback analysis
    from processing.feedback import analyze_and_log_run
    analyze_and_log_run(result)


def run_analysis_only() -> None:
    """Run analysis on existing tweets.json and news.json."""

    result = analyze_documents()
    save_results(result)


def run_full_pipeline() -> None:
    """Placeholder kept for symmetry; scraping is orchestrated in run.py."""

    run_analysis_only()


