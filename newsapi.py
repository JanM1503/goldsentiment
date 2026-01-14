import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List

import requests
from dotenv import load_dotenv


load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NEWS_JSON_PATH = PROJECT_ROOT / "news.json"

# Queries tilted toward gold + macro/politics, split to stay under NewsAPI's
# 500-character limit for the q parameter. We call both and merge results.
NEWS_QUERIES = [
    # Core gold + macro drivers
    (
        "gold OR bullion OR \"gold price\" OR \"gold demand\" OR "
        "\"central bank gold\" OR \"gold reserves\" OR "
        "inflation OR deflation OR recession OR \"interest rates\" OR \"real yields\" OR "
        "\"monetary policy\" OR BRICS OR \"safe haven\" OR geopolitics OR sanctions"
    ),
    # Gold + mining/commodities/BRICS extras
    (
        "gold OR bullion OR \"gold price\" OR \"gold demand\" OR "
        "mining OR miners OR \"gold miner\" OR \"gold miners\" OR "
        "precious metals OR commodity OR commodities OR silver OR platinum OR \"gold ETF\" OR "
        "emerging markets OR \"trade war\" OR tariffs OR embargo OR \"de-dollarization\""
    ),
]

# Additional economic / political keywords used for optional local filtering of
# articles. By default we do *not* enforce this filter so that we fetch a wide
# range of stories; set USE_ECON_FILTER = True to enable it.
# CHANGED: Now enabled by default to filter irrelevant content
USE_ECON_FILTER = True

ECON_POL_KEYWORDS = [
    # Macro & inflation
    "inflation",
    "stagflation",
    "deflation",
    "hyperinflation",
    "recession",
    "slowdown",
    "growth",
    "economic growth",
    "gdp",
    "cpi",
    "ppi",
    "core inflation",
    "unemployment",
    "jobless",
    "labor market",
    "labour market",
    "wage growth",

    # Rates & central banks
    "interest rate",
    "interest rates",
    "rate hike",
    "rate hikes",
    "rate cut",
    "rate cuts",
    "tightening",
    "easing",
    "pivot",
    "central bank",
    "central banks",
    "federal reserve",
    "fed",
    "ecb",
    "bank of england",
    "boj",
    "riksbank",
    "monetary policy",
    "policy meeting",
    "dot plot",
    "forward guidance",

    # Bonds, yields, currencies
    "bond yield",
    "bond yields",
    "treasury yield",
    "treasury yields",
    "yield curve",
    "inverted yield curve",
    "spread widening",
    "credit spread",
    "credit spreads",
    "dollar index",
    "dxy",
    "usd",
    "fx market",
    "currency crisis",
    "de-dollarization",
    "capital flight",

    # Fiscal & debt
    "fiscal policy",
    "budget deficit",
    "deficit",
    "government debt",
    "public debt",
    "debt ceiling",
    "stimulus",
    "bailout",
    "austerity",

    # Markets & risk sentiment
    "stock market",
    "equities",
    "selloff",
    "sell-off",
    "risk-off",
    "risk-on",
    "volatility",
    "vix",
    "market turmoil",
    "market rout",
    "flight to safety",
    "safe haven",

    # Gold & commodities
    "gold",
    "bullion",
    "gold price",
    "gold demand",
    "central bank gold",
    "gold reserves",
    "precious metal",
    "precious metals",
    "commodity",
    "commodities",
    "mining",
    "miners",
    "gold miner",
    "gold miners",
    "silver",
    "platinum",

    # EM / BRICS / geopolitics
    "emerging market",
    "emerging markets",
    "developing market",
    "developing markets",
    "brics",
    "de-dollarization",
    "sanction",
    "sanctions",
    "geopolitic",
    "geopolitics",
    "war",
    "conflict",
    "crisis",
    "trade war",
    "tariff",
    "tariffs",
    "embargo",

    # Political & policy
    "politic",
    "politics",
    "election",
    "elections",
    "government",
    "coalition",
    "parliament",
    "policy",
    "fiscal policy",
    "government spending",
    "budget",
    "stimulus package",
    "regulation",
]


def _get_newsapi_key() -> str:
    key = os.getenv("NEWSAPI_KEY")
    if not key:
        raise RuntimeError("NEWSAPI_KEY is not set in .env")
    return key


# Spam/irrelevant domains and patterns to reject
SPAM_DOMAINS = [
    "dansdeals.com", "redflagdeals.com", "gamespot.com", "disneyfoodblog.com",
    "brobible.com", "sneakernews.com", "highsnobiety.com", "metalsucks.com",
    "wwd.com", "pcgamer.com", "rollingstone.com", "deadline.com",
    "nypost.com/sports", "usatoday.com/story/sports", "cbc.ca/sports",
    "consent.yahoo.com", "fool.com.au", "mccoveychronicles.com",
    "vikingsterritory.com", "dailysignal.com", "techpowerup.com"
]

SPAM_KEYWORDS = [
    "nfl", "nba", "mlb", "nhl", "soccer", "football", "basketball", "baseball",
    "steelers", "lakers", "cowboys", "playoffs", "super bowl", "world series",
    "zodiac", "horoscope", "astrology", "porsche", "ferrari", "lamborghini",
    "adidas", "nike", "sneaker", "fashion", "kitchen", "recipe", "video game",
    "videogame", "gaming", "movie review", "soundtrack", "radiohead", "concert",
    "neon sign", "olympics", "skiing", "snowboard", "sailing", "softball",
    "bluey", "disney", "roku", "streaming", "amazon deal", "prime members",
    "lottery", "gold medal", "olympic gold", "halfpipe"
]

# Gold-specific relevance keywords (tiered)
GOLD_DIRECT = [
    "gold", "bullion", "gold price", "gold demand", "gold miner", "gold mining",
    "gold reserves", "gold etf", "precious metal", "newmont", "barrick"
]

GOLD_MACRO = [
    "federal reserve", "fed", "powell", "interest rate", "rate hike", "rate cut",
    "inflation", "deflation", "recession", "stagflation", "monetary policy",
    "central bank", "treasury yield", "dollar index", "dxy", "safe haven",
    "brics", "de-dollarization", "currency crisis", "debt ceiling"
]

def _calculate_gold_relevance_score(art: Dict[str, Any]) -> float:
    """Calculate gold relevance score from 0-1.
    
    Returns:
        0.0 = completely irrelevant (spam/sports/lifestyle)
        0.3-0.5 = macro-relevant (Fed, inflation, geopolitics)
        0.7-1.0 = directly gold-relevant
    """
    blob = " ".join(
        str(art.get(k) or "") for k in ("title", "description", "content")
    ).lower()
    
    url = str(art.get("url", "")).lower()
    
    # Immediate rejection: spam domains
    if any(domain in url for domain in SPAM_DOMAINS):
        return 0.0
    
    # Immediate rejection: spam keywords (unless gold price/bullion is mentioned)
    # Need to be careful: "gold medal" != "gold price"
    has_gold_financial = any(kw in blob for kw in [
        "gold price", "gold demand", "gold market", "bullion",
        "gold mining", "gold miner", "gold etf", "precious metal"
    ])
    if not has_gold_financial:
        for spam_kw in SPAM_KEYWORDS:
            if spam_kw in blob:
                return 0.0
        # Also reject if it mentions "gold medal" or "olympic" without financial context
        if ("gold medal" in blob or "olympic" in blob) and "gold price" not in blob:
            return 0.0
    
    # Score gold-direct mentions
    gold_direct_score = sum(1 for kw in GOLD_DIRECT if kw in blob)
    if gold_direct_score > 0:
        return min(1.0, 0.7 + (gold_direct_score * 0.1))
    
    # Score macro relevance
    macro_score = sum(1 for kw in GOLD_MACRO if kw in blob)
    if macro_score > 0:
        return min(0.6, 0.3 + (macro_score * 0.1))
    
    # Fallback to old econ/pol keywords - but be more strict
    econ_score = sum(1 for kw in ECON_POL_KEYWORDS if kw in blob)
    if econ_score >= 3:  # Require at least 3 matches (increased from 2)
        return 0.3
    
    return 0.0

def _is_relevant_article(art: Dict[str, Any]) -> bool:
    """Return True if the article looks economic / political enough to matter.

    If USE_ECON_FILTER is False (default), this always returns True and we rely
    on NEWS_QUERIES + FinBERT to judge relevance. If True, we require at least
    one ECON_POL_KEYWORDS hit in title/description/content.
    """

    if not USE_ECON_FILTER:
        return True

    # Use new relevance scoring - require at least 0.3 score
    relevance_score = _calculate_gold_relevance_score(art)
    return relevance_score >= 0.3


def fetch_news(
    max_pages: int = 1,
    page_size: int = 100,
    from_iso: str | None = None,
    to_iso: str | None = None,
) -> List[Dict[str, Any]]:
    """Fetch gold- and macro-related news (paginated) and return article list.

    Notes for NewsAPI Developer accounts:
      - The ``everything`` endpoint is limited to 100 results total.
      - We therefore default to ``max_pages=1`` with ``page_size=100`` to
        avoid 426 Upgrade Required errors. If you upgrade your plan and want
        more depth, you can increase ``max_pages``.

    - Uses up to ``max_pages`` pages with ``page_size`` results each per query
      in NEWS_QUERIES.
    - Deduplicates by URL so the same story is not counted twice per run.
    - Applies a local economic/political relevance filter.
    """

    api_key = _get_newsapi_key()

    normalized: List[Dict[str, Any]] = []
    seen_urls: set[str] = set()

    for q in NEWS_QUERIES:
        for page in range(1, max_pages + 1):
            params = {
                "q": q,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": page_size,
                "page": page,
                "apiKey": api_key,
            }
            if from_iso:
                params["from"] = from_iso
            if to_iso:
                params["to"] = to_iso

        resp = requests.get("https://newsapi.org/v2/everything", params=params, timeout=30)
        try:
            resp.raise_for_status()
        except requests.HTTPError as exc:
            # If the *first* page for a given query fails, surface the error so
            # the user sees why no articles are being fetched. For later pages,
            # just stop and keep what we have so far.
            msg = f"NewsAPI HTTPError on page {page}: {exc} (status={resp.status_code})"
            try:
                detail = resp.json().get("message")
                if detail:
                    msg += f" - {detail}"
            except Exception:
                pass
            print(msg)
            if page == 1:
                raise
            break
        payload = resp.json()

        articles = payload.get("articles", [])
        print(f"NewsAPI query='{q[:60]}...' page={page}: {len(articles)} raw articles")
        if not articles:
            break

        for art in articles:
            # Local relevance filter: skip clearly non-economic / non-political
            # stories even if they match the broad NEWS_QUERY.
            if not _is_relevant_article(art):
                continue

            url = art.get("url")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)

            published_at = art.get("publishedAt")
            ts = None
            if published_at:
                try:
                    ts = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
                except Exception:
                    ts = None
            if ts is None:
                ts_iso = datetime.now(timezone.utc).isoformat()
            else:
                ts_iso = ts.astimezone(timezone.utc).isoformat()

            normalized.append(
                {
                    "title": art.get("title"),
                    "description": art.get("description"),
                    "content": art.get("content"),
                    "url": art.get("url"),
                    "timestamp": ts_iso,
                }
            )

    return normalized


def _load_existing_news(path: Path = NEWS_JSON_PATH) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []


def save_news_to_json(articles: List[Dict[str, Any]], path: Path = NEWS_JSON_PATH) -> None:
    # Sort newest first by timestamp if available.
    def _key(a: Dict[str, Any]) -> str:
        return str(a.get("timestamp", ""))

    articles_sorted = sorted(articles, key=_key, reverse=True)
    path.write_text(json.dumps(articles_sorted, indent=2, ensure_ascii=False), encoding="utf-8")


def run_cli() -> None:
    existing = _load_existing_news(NEWS_JSON_PATH)
    existing_by_url: Dict[str, Dict[str, Any]] = {}
    for art in existing:
        url = art.get("url")
        if url:
            existing_by_url[url] = art

    fetched = fetch_news()
    new_count = 0
    for art in fetched:
        url = art.get("url")
        if not url:
            continue
        if url not in existing_by_url:
            new_count += 1
        existing_by_url[url] = art

    merged = list(existing_by_url.values())
    save_news_to_json(merged, NEWS_JSON_PATH)

    # Log a short summary for this run.
    print(f"NewsAPI: {len(fetched)} fetched, {new_count} new, {len(merged)} total stored.")


def backfill_last_days(total_days: int = 30, window_days: int = 3) -> List[Dict[str, Any]]:
    """Fetch historical news over the last ``total_days`` in rolling windows.

    This walks backwards in time in windows (default 3 days) and calls
    ``fetch_news`` for each window, then concatenates and returns all articles.
    """

    now = datetime.now(timezone.utc)
    all_articles: List[Dict[str, Any]] = []

    for offset in range(0, total_days, window_days):
        end = now - timedelta(days=offset)
        start = now - timedelta(days=min(offset + window_days, total_days))
        batch = fetch_news(from_iso=start.isoformat(), to_iso=end.isoformat())
        all_articles.extend(batch)

    return all_articles


def run_backfill_cli(total_days: int = 30, window_days: int = 3) -> None:
    """CLI helper: backfill news.json with historical news (last 30 days)."""

    existing = _load_existing_news(NEWS_JSON_PATH)
    existing_by_url: Dict[str, Dict[str, Any]] = {}
    for art in existing:
        url = art.get("url")
        if url:
            existing_by_url[url] = art

    fetched = backfill_last_days(total_days=total_days, window_days=window_days)
    new_count = 0
    for art in fetched:
        url = art.get("url")
        if not url:
            continue
        if url not in existing_by_url:
            new_count += 1
        existing_by_url[url] = art

    merged = list(existing_by_url.values())
    save_news_to_json(merged, NEWS_JSON_PATH)
    print(f"NewsAPI backfill: {len(fetched)} fetched, {new_count} new, {len(merged)} total stored.")


if __name__ == "__main__":  # Manual testing
    # Run a single incremental update when executed directly.
    run_cli()
