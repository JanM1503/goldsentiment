"""Feedback analysis system for Gold Sentiment Index.

This module tracks each run of the GSI pipeline and provides detailed analytics
to help diagnose volatility issues and understand what's driving the sentiment.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEEDBACK_PATH = PROJECT_ROOT / "feedback.json"


def _load_existing_feedback() -> List[Dict[str, Any]]:
    """Load existing feedback history."""
    if not FEEDBACK_PATH.exists():
        return []
    try:
        return json.loads(FEEDBACK_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []


def analyze_and_log_run(sentiment_results: Dict[str, Any]) -> None:
    """Analyze a GSI run and append feedback to feedback.json.
    
    Args:
        sentiment_results: The full sentiment_results dict from sentiment.py
    """
    timestamp = sentiment_results.get("timestamp", datetime.now(timezone.utc).isoformat())
    gsi = sentiment_results.get("gsi", 0.0)
    classification = sentiment_results.get("classification", "Unknown")
    
    news_data = sentiment_results.get("news", {})
    documents = news_data.get("documents", [])
    
    # Statistics
    total_articles = len(documents)
    
    # Find most positive and negative articles
    sorted_by_positive = sorted(
        documents,
        key=lambda d: d.get("positive", 0.0),
        reverse=True
    )
    sorted_by_negative = sorted(
        documents,
        key=lambda d: d.get("negative", 0.0),
        reverse=True
    )
    
    top_positive = [
        {
            "title": d.get("text", "")[:120],
            "score": round(d.get("positive", 0.0), 3),
            "url": d.get("id", "")
        }
        for d in sorted_by_positive[:5]
    ]
    
    top_negative = [
        {
            "title": d.get("text", "")[:120],
            "score": round(d.get("negative", 0.0), 3),
            "url": d.get("id", "")
        }
        for d in sorted_by_negative[:5]
    ]
    
    # Calculate sentiment distribution
    avg_positive = sum(d.get("positive", 0.0) for d in documents) / max(total_articles, 1)
    avg_negative = sum(d.get("negative", 0.0) for d in documents) / max(total_articles, 1)
    avg_neutral = sum(d.get("neutral", 0.0) for d in documents) / max(total_articles, 1)
    
    # Count extreme articles (very positive or very negative)
    extreme_positive = sum(1 for d in documents if d.get("positive", 0.0) > 0.7)
    extreme_negative = sum(1 for d in documents if d.get("negative", 0.0) > 0.7)
    
    # Count articles with clear directional sentiment
    clear_bullish = sum(1 for d in documents if d.get("positive", 0.0) - d.get("negative", 0.0) > 0.3)
    clear_bearish = sum(1 for d in documents if d.get("negative", 0.0) - d.get("positive", 0.0) > 0.3)
    
    # Check for gold-related keywords in articles
    from scraping.newsapi import GOLD_DIRECT, GOLD_MACRO
    
    gold_direct_count = 0
    gold_macro_count = 0
    
    for doc in documents:
        text = doc.get("text", "").lower()
        if any(kw in text for kw in GOLD_DIRECT):
            gold_direct_count += 1
        elif any(kw in text for kw in GOLD_MACRO):
            gold_macro_count += 1
    
    # Create feedback entry
    feedback_entry = {
        "timestamp": timestamp,
        "gsi_value": round(gsi, 2),
        "classification": classification,
        "statistics": {
            "total_articles": total_articles,
            "gold_direct_articles": gold_direct_count,
            "gold_macro_articles": gold_macro_count,
            "other_articles": total_articles - gold_direct_count - gold_macro_count,
            "avg_positive": round(avg_positive, 3),
            "avg_negative": round(avg_negative, 3),
            "avg_neutral": round(avg_neutral, 3),
            "extreme_positive": extreme_positive,
            "extreme_negative": extreme_negative,
            "clear_bullish": clear_bullish,
            "clear_bearish": clear_bearish,
        },
        "top_5_most_positive": top_positive,
        "top_5_most_negative": top_negative,
        "analysis": _generate_analysis(
            gsi, total_articles, gold_direct_count, 
            extreme_negative, clear_bullish, clear_bearish,
            avg_positive, avg_negative
        )
    }
    
    # Load existing feedback and append
    feedback_history = _load_existing_feedback()
    feedback_history.append(feedback_entry)
    
    # Keep only last 30 runs to avoid file bloat
    feedback_history = feedback_history[-30:]
    
    # Save
    FEEDBACK_PATH.write_text(
        json.dumps(feedback_history, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    
    print(f"\n{'='*70}")
    print(f"GSI RUN ANALYSIS - {timestamp[:19]}")
    print(f"{'='*70}")
    print(f"GSI Value: {round(gsi, 2)} ({classification})")
    print(f"\nArticles Analyzed: {total_articles}")
    print(f"  - Gold Direct: {gold_direct_count}")
    print(f"  - Macro-related: {gold_macro_count}")
    print(f"  - Other: {total_articles - gold_direct_count - gold_macro_count}")
    print(f"\nSentiment Distribution:")
    print(f"  - Avg Positive: {round(avg_positive, 3)}")
    print(f"  - Avg Negative: {round(avg_negative, 3)}")
    print(f"  - Clear Bullish: {clear_bullish} articles")
    print(f"  - Clear Bearish: {clear_bearish} articles")
    print(f"\nANALYSIS:")
    for line in feedback_entry["analysis"]:
        print(f"  • {line}")
    print(f"{'='*70}\n")
    print(f"Full feedback saved to: {FEEDBACK_PATH}")


def _generate_analysis(
    gsi: float,
    total: int,
    gold_direct: int,
    extreme_neg: int,
    clear_bull: int,
    clear_bear: int,
    avg_pos: float,
    avg_neg: float
) -> List[str]:
    """Generate human-readable analysis of the run."""
    analysis = []
    
    # Check article relevance
    gold_ratio = gold_direct / max(total, 1)
    if gold_ratio < 0.2:
        analysis.append(f"⚠️  LOW GOLD RELEVANCE: Only {gold_direct}/{total} ({gold_ratio*100:.0f}%) articles directly mention gold")
    elif gold_ratio > 0.5:
        analysis.append(f"✓ HIGH GOLD RELEVANCE: {gold_direct}/{total} ({gold_ratio*100:.0f}%) articles directly mention gold")
    
    # Check for extreme negative influence
    if extreme_neg > 3:
        analysis.append(f"⚠️  {extreme_neg} highly negative articles detected - may be dragging index down")
    
    # Check sentiment balance
    net_sentiment = avg_pos - avg_neg
    if net_sentiment > 0.1:
        analysis.append(f"✓ Net positive sentiment: {net_sentiment:.3f} (bullish signal)")
    elif net_sentiment < -0.1:
        analysis.append(f"⚠️  Net negative sentiment: {net_sentiment:.3f} (bearish signal)")
    else:
        analysis.append(f"Neutral sentiment: {net_sentiment:.3f}")
    
    # Check GSI alignment with market
    if gsi < 50 and clear_bull > clear_bear:
        analysis.append("⚠️  GSI shows bearish but more bullish than bearish articles - possible miscalibration")
    elif gsi > 70 and clear_bear > clear_bull:
        analysis.append("⚠️  GSI shows bullish but more bearish than bullish articles - possible miscalibration")
    
    # Expected range comment
    if gsi < 40:
        analysis.append("GSI < 40: Market in fear mode - typically occurs during gold selloffs or strong USD")
    elif gsi > 70:
        analysis.append("GSI > 70: Market in greed mode - typical during gold bull runs")
    else:
        analysis.append("GSI 40-70: Neutral to mildly bullish range")
    
    return analysis
