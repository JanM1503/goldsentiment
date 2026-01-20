from __future__ import annotations

import sys
from pathlib import Path

from scraping.newsapi import run_cli as news_cli
from processing.sentiment import run_analysis_only
from processing.dashboard import generate_dashboard
from processing.gold_price import display_gold_price


PROJECT_ROOT = Path(__file__).resolve().parent


USAGE = """Usage:
  python run.py sentiment update     # fetch NewsAPI, analyze, update JSON + dashboard
  python run.py sentiment news       # fetch NewsAPI only
  python run.py sentiment analyze    # run FinBERT on existing news.json + update dashboard
  python run.py sentiment dashboard  # regenerate dashboard.html from existing sentiment_results.json
  python run.py gold price           # show current gold price and % changes (1d, 7d, 30d)
"""


def cmd_update() -> None:
    # Fetch News, then analyze
    news_cli()
    run_analysis_only()
    generate_dashboard()


def cmd_news() -> None:
    news_cli()


def cmd_analyze() -> None:
    run_analysis_only()
    generate_dashboard()


def cmd_dashboard_only() -> None:
    """Regenerate dashboard.html from the latest sentiment_results.json.

    This does NOT fetch new data or rerun FinBERT; it only redraws the gauge
    using whatever is currently stored in sentiment_results.json.
    """

    generate_dashboard()


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    if len(argv) < 1:
        print(USAGE)
        return
    
    # Handle 'gold price' command
    if argv[0] == "gold" and len(argv) >= 2 and argv[1] == "price":
        display_gold_price()
        return

    if len(argv) < 2 or argv[0] != "sentiment":
        print(USAGE)
        return

    sub = argv[1]
    if sub == "update":
        cmd_update()
    elif sub == "news":
        cmd_news()
    elif sub == "analyze":
        cmd_analyze()
    elif sub == "dashboard":
        cmd_dashboard_only()
    else:
        print(USAGE)


if __name__ == "__main__":
    main()
