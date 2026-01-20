"""
Gold Price Fetcher - OANDA API

Fetches current gold price (XAU/USD) from OANDA and calculates
percentage changes over 1 day, 7 days, and 30 days.
"""

import requests
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional


# OANDA API Configuration
OANDA_API_KEY = "7853cf1199152030d7cfed27ff0dfadd-e141ec058dafd6ca2090eb0887b3791d"
OANDA_BASE_URL = "https://api-fxpractice.oanda.com/v3"
INSTRUMENT = "XAU_USD"  # Gold vs US Dollar


def fetch_oanda_candles(days_back: int = 31) -> Optional[list]:
    """
    Fetch historical candlestick data from OANDA.
    
    Args:
        days_back: Number of days of historical data to fetch
    
    Returns:
        List of candles or None if request fails
    """
    try:
        url = f"{OANDA_BASE_URL}/instruments/{INSTRUMENT}/candles"
        
        headers = {
            "Authorization": f"Bearer {OANDA_API_KEY}",
            "Content-Type": "application/json"
        }
        
        params = {
            "granularity": "D",  # Daily candles
            "count": days_back + 1  # Get enough data
        }
        
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        return data.get("candles", [])
    
    except Exception as e:
        print(f"⚠️  Error fetching OANDA data: {e}")
        return None


def calculate_gold_price_data() -> Optional[Dict[str, float]]:
    """
    Fetch gold price and calculate percentage changes.
    
    Returns:
        Dict with:
        - current_price: Current gold price
        - change_1d_pct: 1-day % change
        - change_7d_pct: 7-day % change
        - change_30d_pct: 30-day % change
    """
    candles = fetch_oanda_candles(days_back=31)
    
    if not candles or len(candles) < 2:
        return None
    
    # Get closing prices
    prices = [float(c["mid"]["c"]) for c in candles if c.get("complete")]
    
    if len(prices) < 2:
        return None
    
    # Current price (most recent)
    current_price = prices[-1]
    
    # Calculate 1-day change
    if len(prices) >= 2:
        price_1d_ago = prices[-2]
        change_1d_pct = ((current_price / price_1d_ago) - 1) * 100
    else:
        change_1d_pct = 0.0
    
    # Calculate 7-day change
    if len(prices) >= 8:
        price_7d_ago = prices[-8]
        change_7d_pct = ((current_price / price_7d_ago) - 1) * 100
    else:
        change_7d_pct = 0.0
    
    # Calculate 30-day change
    if len(prices) >= 31:
        price_30d_ago = prices[-31]
        change_30d_pct = ((current_price / price_30d_ago) - 1) * 100
    else:
        change_30d_pct = 0.0
    
    return {
        "current_price": current_price,
        "change_1d_pct": change_1d_pct,
        "change_7d_pct": change_7d_pct,
        "change_30d_pct": change_30d_pct,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "OANDA (XAU/USD)"
    }


def format_change(change_pct: float) -> str:
    """Format percentage change with sign."""
    if change_pct > 0:
        return f"+{change_pct:.2f}%"
    else:
        return f"{change_pct:.2f}%"


def display_gold_price() -> None:
    """Display current gold price and changes."""
    print("\n" + "=" * 70)
    print("📊 GOLD PRICE (XAU/USD)")
    print("=" * 70)
    
    data = calculate_gold_price_data()
    
    if data is None:
        print("\n❌ Unable to fetch gold price from OANDA")
        print("\nPossible issues:")
        print("  - Internet connection")
        print("  - OANDA API key invalid")
        print("  - Rate limit exceeded")
        print("=" * 70 + "\n")
        return
    
    current = data["current_price"]
    change_1d = data["change_1d_pct"]
    change_7d = data["change_7d_pct"]
    change_30d = data["change_30d_pct"]
    
    # Display
    print(f"\n{'Current Price:':<20} ${current:,.2f} per oz")
    print(f"{'Data Source:':<20} {data['source']}")
    print("-" * 70)
    print(f"{'1-Day Change:':<20} {format_change(change_1d):<15} {'🟢' if change_1d > 0 else '🔴' if change_1d < 0 else '⚪'}")
    print(f"{'7-Day Change:':<20} {format_change(change_7d):<15} {'🟢' if change_7d > 0 else '🔴' if change_7d < 0 else '⚪'}")
    print(f"{'30-Day Change:':<20} {format_change(change_30d):<15} {'🟢' if change_30d > 0 else '🔴' if change_30d < 0 else '⚪'}")
    
    # Interpretation
    print("\n" + "-" * 70)
    print("Interpretation:")
    
    if change_30d > 5:
        print("  📈 Strong bullish trend (30d: >+5%)")
    elif change_30d > 2:
        print("  📈 Moderate bullish trend (30d: +2% to +5%)")
    elif change_30d > -2:
        print("  ➡️  Neutral/sideways (30d: -2% to +2%)")
    elif change_30d > -5:
        print("  📉 Moderate bearish trend (30d: -2% to -5%)")
    else:
        print("  📉 Strong bearish trend (30d: <-5%)")
    
    # Timestamp
    timestamp = datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))
    print(f"\nLast updated: {timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    display_gold_price()
