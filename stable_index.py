"""
Stabilized Gold Sentiment Index (GSI) - Fear & Greed Style

This module implements a slow-moving, price-anchored sentiment barometer
that dampens volatility and prevents unrealistic index values.

Key Features:
- Exponential Moving Average (EMA) for temporal smoothing
- Price-based reality anchoring (1d, 7d, 30d gold performance)
- Sentiment-price consistency checks
- Maximum change rate limiting (inertia)
- Multi-horizon sentiment aggregation
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GSI_HISTORY_PATH = PROJECT_ROOT / "docs" / "gsi_history.json"


# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

# Smoothing parameters (higher = slower movement)
EMA_ALPHA = 0.15  # How much new data affects index (0.1-0.3 typical for Fear & Greed)
                   # 0.15 means: 15% new value, 85% old value

# Maximum change per update (prevents wild swings)
MAX_CHANGE_PER_UPDATE = 5.0  # Index can change max ±5 points per update

# Sentiment dampening (reduces raw sentiment volatility)
SENTIMENT_DAMPENING = 0.6  # Scale raw sentiment by 60% before applying

# Price anchoring weights (how much price influences final index)
PRICE_ANCHOR_WEIGHT = 0.3  # 30% price, 70% sentiment

# Consistency penalty (penalize sentiment that contradicts price)
CONSISTENCY_PENALTY_STRENGTH = 0.4  # How strongly to penalize inconsistency

# Neutral zone widening (pull extreme values toward center)
NEUTRAL_GRAVITY = 0.15  # Subtle pull toward 50


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class PriceData:
    """Gold price performance across multiple horizons."""
    current_price: float
    change_1d_pct: float
    change_7d_pct: float
    change_30d_pct: float
    timestamp: str


@dataclass
class StabilizedIndexComponents:
    """All components of the stabilized GSI calculation."""
    # Raw sentiment from news
    raw_sentiment: float  # [0, 100]
    
    # Price-based components
    price_momentum: float  # [0, 100] derived from price changes
    
    # Dampened sentiment
    dampened_sentiment: float  # [0, 100]
    
    # Sentiment-price consistency score
    consistency_score: float  # [0, 1], 1 = perfectly consistent
    
    # Previous GSI value (for EMA)
    previous_gsi: float  # [0, 100]
    
    # Final smoothed GSI
    final_gsi: float  # [0, 100]
    
    # Classification
    classification: str
    
    # Metadata
    change_from_previous: float
    price_data: Optional[PriceData]


# ============================================================================
# PRICE DATA FETCHING
# ============================================================================

def fetch_gold_price_data() -> Optional[PriceData]:
    """
    Fetch current gold price and historical performance using OANDA API.
    
    Returns:
        PriceData with 1d, 7d, 30d percentage changes, or None if unavailable
    """
    try:
        # Import the gold price module
        from processing.gold_price import calculate_gold_price_data
        
        # Fetch data from OANDA
        data = calculate_gold_price_data()
        
        if data is None:
            return None
        
        # Convert to PriceData format
        return PriceData(
            current_price=data["current_price"],
            change_1d_pct=data["change_1d_pct"],
            change_7d_pct=data["change_7d_pct"],
            change_30d_pct=data["change_30d_pct"],
            timestamp=data["timestamp"]
        )
    
    except Exception as e:
        print(f"Warning: Could not fetch gold price data: {e}")
        return None


def calculate_price_momentum(price_data: PriceData) -> float:
    """
    Convert price performance into a momentum score [0, 100].
    
    Logic:
    - Weighted average of 1d, 7d, 30d changes
    - 30d gets most weight (medium-term trend)
    - Normalized to [0, 100] scale
    
    Examples:
        +5% across all horizons → ~75 (bullish)
        -5% across all horizons → ~25 (bearish)
        Mixed signals → ~50 (neutral)
    """
    # Weights: favor medium-term (30d) over short-term noise
    w1d = 0.2
    w7d = 0.3
    w30d = 0.5
    
    # Weighted average of price changes
    weighted_change = (
        price_data.change_1d_pct * w1d +
        price_data.change_7d_pct * w7d +
        price_data.change_30d_pct * w30d
    )
    
    # Normalize to [0, 100]
    # Assume ±10% is extreme (maps to 0 or 100)
    # 0% maps to 50
    normalized = 50 + (weighted_change / 10.0) * 50
    
    # Clip to valid range
    return max(0.0, min(100.0, normalized))


# ============================================================================
# SENTIMENT-PRICE CONSISTENCY
# ============================================================================

def calculate_consistency_score(
    sentiment: float,
    price_momentum: float
) -> float:
    """
    Measure how consistent sentiment is with price action.
    
    Returns consistency score [0, 1]:
        1.0 = perfect consistency (sentiment matches price)
        0.5 = neutral/uncorrelated
        0.0 = strong contradiction
    
    Logic:
        - If gold is up +10% (price_momentum=75) but sentiment is 20 → low consistency
        - If gold is flat (price_momentum=50) and sentiment is 50 → perfect consistency
        - If gold is down -5% (price_momentum=35) and sentiment is 30 → high consistency
    """
    # Calculate absolute difference
    diff = abs(sentiment - price_momentum)
    
    # Normalize: 0 diff = 1.0 consistency, 50 diff = 0.0 consistency
    consistency = 1.0 - (diff / 50.0)
    
    return max(0.0, min(1.0, consistency))


def apply_consistency_penalty(
    sentiment: float,
    price_momentum: float,
    consistency_score: float,
    penalty_strength: float = CONSISTENCY_PENALTY_STRENGTH
) -> float:
    """
    Adjust sentiment toward price momentum if they're inconsistent.
    
    This prevents sentiment from being wildly detached from reality.
    """
    if consistency_score > 0.7:
        # Already consistent, no penalty
        return sentiment
    
    # Calculate penalty amount (proportional to inconsistency)
    inconsistency = 1.0 - consistency_score
    penalty_amount = inconsistency * penalty_strength
    
    # Pull sentiment toward price momentum
    adjusted = sentiment * (1 - penalty_amount) + price_momentum * penalty_amount
    
    return adjusted


# ============================================================================
# TEMPORAL SMOOTHING
# ============================================================================

def load_previous_gsi() -> float:
    """Load the most recent GSI value from history for EMA calculation."""
    try:
        if not GSI_HISTORY_PATH.exists():
            return 50.0  # Default neutral
        
        history = json.loads(GSI_HISTORY_PATH.read_text(encoding="utf-8"))
        if not history or len(history) == 0:
            return 50.0
        
        # Get most recent entry
        latest = history[-1]
        return float(latest.get("gsi", 50.0))
    
    except Exception as e:
        print(f"Warning: Could not load previous GSI: {e}")
        return 50.0


def save_gsi_to_history(gsi: float, timestamp: str, components: Dict[str, Any]) -> None:
    """Append current GSI to historical record."""
    try:
        history = []
        if GSI_HISTORY_PATH.exists():
            history = json.loads(GSI_HISTORY_PATH.read_text(encoding="utf-8"))
        
        history.append({
            "timestamp": timestamp,
            "gsi": gsi,
            "components": components
        })
        
        # Keep only last 90 days
        cutoff = datetime.now(timezone.utc) - timedelta(days=90)
        history = [
            h for h in history
            if datetime.fromisoformat(h["timestamp"].replace("Z", "+00:00")) > cutoff
        ]
        
        GSI_HISTORY_PATH.write_text(
            json.dumps(history, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
    
    except Exception as e:
        print(f"Warning: Could not save GSI history: {e}")


def apply_ema_smoothing(
    new_value: float,
    previous_value: float,
    alpha: float = EMA_ALPHA
) -> float:
    """
    Apply Exponential Moving Average smoothing.
    
    Formula: EMA = α * new_value + (1 - α) * previous_value
    
    With α=0.15:
        - New value contributes 15%
        - Previous value contributes 85%
        - Requires ~20 updates to fully transition
    """
    return alpha * new_value + (1 - alpha) * previous_value


def apply_change_rate_limit(
    new_value: float,
    previous_value: float,
    max_change: float = MAX_CHANGE_PER_UPDATE
) -> float:
    """
    Limit how much the index can change in a single update.
    
    This prevents sudden jumps even if EMA allows them.
    """
    change = new_value - previous_value
    
    if abs(change) <= max_change:
        return new_value
    
    # Limit the change
    if change > 0:
        return previous_value + max_change
    else:
        return previous_value - max_change


# ============================================================================
# NEUTRAL GRAVITY
# ============================================================================

def apply_neutral_gravity(value: float, gravity: float = NEUTRAL_GRAVITY) -> float:
    """
    Apply subtle pull toward 50 (neutral zone).
    
    This prevents the index from getting stuck at extremes and
    makes it easier to return to neutral when news is mixed.
    """
    # Calculate pull toward 50
    distance_from_neutral = value - 50.0
    pull = distance_from_neutral * gravity
    
    return value - pull


# ============================================================================
# MAIN CALCULATION
# ============================================================================

def compute_stabilized_index(
    raw_sentiment: float,  # [0, 100] from news analysis
    price_data: Optional[PriceData] = None
) -> StabilizedIndexComponents:
    """
    Compute stabilized GSI with all dampening mechanisms.
    
    Pipeline:
        1. Fetch price data (if available)
        2. Calculate price momentum
        3. Dampen raw sentiment
        4. Check sentiment-price consistency
        5. Apply consistency penalty if needed
        6. Blend sentiment + price with weights
        7. Apply neutral gravity
        8. Apply EMA smoothing with previous value
        9. Apply rate-of-change limit
        10. Classify final value
    
    Args:
        raw_sentiment: Unsmoothed sentiment from news [0, 100]
        price_data: Optional gold price performance data
    
    Returns:
        StabilizedIndexComponents with all intermediate values
    """
    # Step 1: Load previous GSI for EMA
    previous_gsi = load_previous_gsi()
    
    # Step 2: Dampen raw sentiment (reduce volatility)
    dampened_sentiment = 50 + (raw_sentiment - 50) * SENTIMENT_DAMPENING
    
    # Step 3: Calculate price momentum (if data available)
    if price_data is not None:
        price_momentum = calculate_price_momentum(price_data)
        
        # Step 4: Check consistency
        consistency_score = calculate_consistency_score(
            dampened_sentiment,
            price_momentum
        )
        
        # Step 5: Apply consistency penalty
        adjusted_sentiment = apply_consistency_penalty(
            dampened_sentiment,
            price_momentum,
            consistency_score
        )
        
        # Step 6: Blend sentiment + price
        blended = (
            adjusted_sentiment * (1 - PRICE_ANCHOR_WEIGHT) +
            price_momentum * PRICE_ANCHOR_WEIGHT
        )
    else:
        # No price data: use sentiment only
        price_momentum = 50.0
        consistency_score = 1.0
        blended = dampened_sentiment
    
    # Step 7: Apply neutral gravity (subtle pull toward 50)
    gravitized = apply_neutral_gravity(blended)
    
    # Step 8: Apply EMA smoothing
    smoothed = apply_ema_smoothing(gravitized, previous_gsi)
    
    # Step 9: Apply rate-of-change limit
    final_gsi = apply_change_rate_limit(smoothed, previous_gsi)
    
    # Step 10: Classify
    classification = _classify_gsi(final_gsi)
    
    # Calculate change
    change = final_gsi - previous_gsi
    
    return StabilizedIndexComponents(
        raw_sentiment=raw_sentiment,
        price_momentum=price_momentum,
        dampened_sentiment=dampened_sentiment,
        consistency_score=consistency_score,
        previous_gsi=previous_gsi,
        final_gsi=final_gsi,
        classification=classification,
        change_from_previous=change,
        price_data=price_data
    )


def _classify_gsi(gsi: float) -> str:
    """Map GSI value into sentiment regime buckets."""
    if gsi < 25:
        return "Extremely Bearish"
    if gsi < 45:
        return "Bearish"
    if gsi < 55:
        return "Neutral"
    if gsi < 75:
        return "Bullish"
    return "Extremely Bullish"


# ============================================================================
# INTEGRATION HELPERS
# ============================================================================

def convert_legacy_sentiment_to_stabilized(
    legacy_gsi: float,
    timestamp: str
) -> Dict[str, Any]:
    """
    Convert legacy GSI calculation to stabilized version.
    
    This is the integration point with your existing code.
    """
    # Fetch price data
    price_data = fetch_gold_price_data()
    
    # Compute stabilized index
    components = compute_stabilized_index(legacy_gsi, price_data)
    
    # Save to history
    save_gsi_to_history(
        components.final_gsi,
        timestamp,
        {
            "raw_sentiment": components.raw_sentiment,
            "dampened_sentiment": components.dampened_sentiment,
            "price_momentum": components.price_momentum,
            "consistency_score": components.consistency_score,
            "change": components.change_from_previous
        }
    )
    
    # Return result in format compatible with existing system
    return {
        "timestamp": timestamp,
        "gsi": components.final_gsi,
        "classification": components.classification,
        "change_from_previous": components.change_from_previous,
        "components": {
            "raw_sentiment": components.raw_sentiment,
            "dampened_sentiment": components.dampened_sentiment,
            "price_momentum": components.price_momentum,
            "consistency_score": components.consistency_score,
            "previous_gsi": components.previous_gsi
        },
        "price_data": {
            "current_price": price_data.current_price if price_data else None,
            "change_1d_pct": price_data.change_1d_pct if price_data else None,
            "change_7d_pct": price_data.change_7d_pct if price_data else None,
            "change_30d_pct": price_data.change_30d_pct if price_data else None
        } if price_data else None
    }
