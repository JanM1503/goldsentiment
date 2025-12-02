# Gold Sentiment Index (GSI)

The **Gold Sentiment Index (GSI)** is a small research project that turns
financial and macro news into a single **0–100 sentiment index** for gold:

- **0 = Extremely Bearish** about gold
- **50 = Neutral**
- **100 = Extremely Bullish** / very bullish tone

It does this by:

1. **Fetching news articles** about gold, macroeconomics, and geopolitics via **NewsAPI**.
2. **Running a financial-domain sentiment model (FinBERT)** over each article.
3. **Weighting** each article by **recency** and **macro impact** (Fed, crises, BRICS, etc.).
4. Aggregating everything into a **Gold Sentiment Index** and
5. Generating a clean **HTML dashboard** with a gauge (0–100) and current regime
   label ("Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed").

The goal: give a quick, model-based view of how **recent news flow** is leaning
for gold (fear vs. greed), similar in spirit to CNN's Fear & Greed Index but
focused on gold and macro headlines.

---

## Project layout

Main files and what they do:

- `run.py`
  - Command-line entry point.
  - Orchestrates fetching news, running sentiment analysis, and building the dashboard.

- `scraping/newsapi.py`
  - Talks to **NewsAPI** (https://newsapi.org/).
  - Builds a set of queries focused on **gold**, **macro**, **rates**, **BRICS**,
    and **geopolitics**.
  - Fetches articles, normalizes them, and saves them to `news.json` in the
    project root.

- `processing/sentiment.py`
  - Loads articles from `news.json`.
  - Uses FinBERT to compute **positive/negative/neutral** scores per article.
  - Applies **recency weights** and **macro impact weights** to each article.
  - Calls `processing/index_calc.py` to turn those into a single GSI value.
  - Saves a structured summary to `sentiment_results.json` and a
    compact snapshot to `gsi_value.json`.

- `processing/index_calc.py`
  - Combines all article-level scores into a single **Gold Sentiment Index**.
  - Maps raw scores into both:
    - `nw`  – net sentiment in [-1, 1]
    - `nw_norm` / `gsi` – the index in [0, 100]
- Classifies the index into 5 regimes:
    - 0–25   → Extremely Bearish
    - 25–45  → Bearish
    - 45–55  → Neutral
    - 55–75  → Bullish
    - 75–100 → Extremely Bullish

- `processing/dashboard.py`
  - Reads `sentiment_results.json`.
- Generates an **interactive HTML dashboard** at `dashboard.html` with:
    - A custom gauge (0–100) colored by regime bands.
    - Current numerical GSI value.
    - Current regime label (Extremely Bearish / Bearish / Neutral / Bullish / Extremely Bullish).
    - Timestamp of the last update.

- `models/finbert_gold.py`
  - Wraps the **FinBERT financial sentiment model** (`yiyanghkust/finbert-tone`).
  - Provides a simple API:
    - `analyze_text(text) -> SentimentScores`
    - `analyze_batch(texts) -> List[SentimentScores]`

- Data files (generated when you run the pipeline):
  - `news.json`
    - All fetched articles (deduplicated by URL), sorted newest → oldest.
  - `sentiment_results.json`
    - Full analysis result (per-article scores plus index components).
  - `gsi_value.json`
    - Minimal payload with just the latest GSI value and classification.
  - `dashboard.html`
    - Self-contained dashboard you can open in any browser.

> Note: `news_new.json` may exist from older runs, but it is **no longer used**.
> The pipeline now only relies on `news.json`. You can safely delete
> `news_new.json` if you want a clean workspace.

---

## Dependencies and requirements

- **Python:** 3.10 or newer is recommended.
- **Packages:** listed in `requirements.txt`:
  - `requests`
  - `torch`
  - `transformers`
  - `pandas`
  - `numpy`
  - `python-dotenv`
  - `peft`
  - `pytz`

You will also need:

- A **NewsAPI account** (free tier is enough to experiment).
- A machine that can run PyTorch. CPU is fine; GPU will just be faster.

---

## Installation (step-by-step)

These steps assume a fresh clone of the repository.

### 1. Clone the repository

```bash
git clone <your-fork-or-repo-url>
cd gold-sentiment-index-main
```

> If you just downloaded a ZIP from GitHub, unzip it and `cd` into the project
> folder instead.


### 2. Install Python dependencies

From inside the project folder with the virtualenv activated:

```bash
pip install -r requirements.txt
# or, equivalently (explicit):
# pip install requests torch transformers pandas numpy python-dotenv peft pytz
```

This will install PyTorch, Transformers, `pytz`, and the helper libraries.

---

## API keys & configuration

The **only API key** required for this project is the **NewsAPI key**.

### 1. Get a NewsAPI key

1. Go to https://newsapi.org/
2. Sign up for a (free) account.
3. Copy your **API key** from the dashboard.

### 2. Edit the existing `.env` file

A template `.env` file is already included in the project root. Open `.env`
in any text editor and replace the placeholder value:

```ini
NEWSAPI_KEY=your_real_newsapi_key_here
```

with your real NewsAPI key, so it looks like:

```ini
NEWSAPI_KEY=abcdef1234567890
```

- Do **not** put quotes around the key.
- This key is loaded by `python-dotenv` inside `scraping/newsapi.py` via
  the environment variable `NEWSAPI_KEY`.
- There is **no hard-coded API key** in the source code; everything goes
  through the environment.

If `NEWSAPI_KEY` is missing or still set to the placeholder, the code will
raise a clear error:

> `RuntimeError: NEWSAPI_KEY is not set in .env`

---

## How to run the project

The main entry point is `run.py`, which exposes a small CLI under the
`sentiment` command.

From the project root (with your virtualenv active):

```bash
python run.py sentiment update
```

> Note: the FinBERT sentiment step can take from a few seconds up to a few
> minutes, depending on how many articles are in `news.json` and how fast your
> CPU/GPU is (first run is usually slower because the model is downloaded).

### CLI commands

`run.py` supports three subcommands:

1. **Fetch news + analyze + update dashboard**

   ```bash
   python run.py sentiment update
   ```

   - Calls `scraping.newsapi.run_cli()` to fetch and merge new articles into `news.json`.
   - Runs FinBERT sentiment analysis over all **recent** articles.
   - Computes the Gold Sentiment Index.
   - Writes `sentiment_results.json`, `gsi_value.json`.
   - Regenerates `dashboard.html` with the latest GSI value.

2. **Fetch news only (no analysis)**

   ```bash
   python run.py sentiment news
   ```

   - Fetches news from NewsAPI using the preconfigured queries.
   - Merges with existing `news.json`, deduplicating by URL.
   - Does **not** run FinBERT or touch the dashboard.

3. **Analyze existing news only (no new API calls)**

   ```bash
   python run.py sentiment analyze
   ```

   - Reads existing `news.json`.
   - Runs FinBERT sentiment on all recent articles.
   - Produces `sentiment_results.json`, `gsi_value.json`.
   - Regenerates `dashboard.html`.

4. **Redraw dashboard only (no new analysis)**

   ```bash
   python run.py sentiment dashboard
   ```

   - Does **not** fetch new data.
   - Does **not** rerun FinBERT.
   - Simply regenerates `dashboard.html` from the latest `sentiment_results.json`.

This is useful if you want to:

- Avoid hitting the NewsAPI quota repeatedly.
- Re-run the analysis after tweaking the model logic, weighting, or index
  computation without fetching new data.

---

## Viewing the dashboard

After running either:

- `python run.py sentiment update`, or
- `python run.py sentiment analyze`

You should see a file called `dashboard.html` in the project root.

Open it in your browser:

- Double-click it in your file explorer, or
- Right-click → "Open with" → choose your browser, or
- From a terminal:
  - macOS: `open dashboard.html`
  - Linux: `xdg-open dashboard.html`
  - Windows (PowerShell): `start dashboard.html`

The dashboard shows:

- A **gauge** from 0 to 100 with color bands:
  - Extremely Bearish (0–25): red
  - Bearish (25–45): light red
  - Neutral (45–55): gray
  - Bullish (55–75): light green
  - Extremely Bullish (75–100): green
- The **current GSI value** (rounded integer) below the gauge.
- The **classification label** (e.g., "Greed").
- The **last updated timestamp** (converted to your local time zone).

---

## What exactly the scripts do (end-to-end)

This section walks through the full pipeline in plain language.

### 1. News collection (`scraping/newsapi.py`)

- Builds multiple **big search queries** aimed at capturing:
  - Gold and precious metals ("gold", "bullion", "gold price", "gold demand").
  - Monetary policy and rates ("Fed", "interest rates", "rate hike", "rate cut").
  - Macro conditions ("inflation", "recession", "deflation", "stagflation").
  - BRICS, sanctions, geopolitics, trade wars, etc.
- Calls the NewsAPI `everything` endpoint with:
  - `language=en`
  - `sortBy=publishedAt`
  - `pageSize=100` (max results per page)
  - `page=1 .. max_pages` (note that free plans are limited; defaults are set
    to stay within the 100-result limit).
- For each article returned, it normalizes into a compact shape:

  ```json
  {
    "title": "...",
    "description": "...",
    "content": "...",
    "url": "https://example.com/...",
    "timestamp": "2025-12-01T08:49:45+00:00"  // always UTC ISO8601
  }
  ```

- Applies an optional **economic/political keyword filter** (disabled by default
  for broader coverage).
- Deduplicates articles by **URL** across runs so the same article is only
  stored once.
- Merges the new articles with existing ones and writes the updated list to
  `news.json` (newest first).

### 2. Document-level sentiment (`processing/sentiment.py` + `models/finbert_gold.py`)

For each article in `news.json`:

1. It creates a **text blob** from:
   - `title`
   - `description`
   - `content`

2. It computes a **recency weight** based on the article timestamp:

   - 0–1 days old   → 1.0 (very fresh, full impact)
   - 1–3 days       → 0.8
   - 3–7 days       → 0.6
   - 7–14 days      → 0.3
   - 14–30 days     → 0.1
   - older than 30d → 0.0 (effectively ignored)

3. It runs the FinBERT sentiment model on the text and gets probabilities:

   - `positive` in [0, 1]
   - `negative` in [0, 1]
   - `neutral` in [0, 1]

4. It computes an **impact weight** per document that:

   - Starts from the **margin** `|positive - negative|` → confidence that the
     headline is clearly bullish or bearish.
   - Boosts the weight by **3x** if certain macro keywords appear, e.g.:
     - "Powell", "Federal Reserve", "rate hike", "recession", "crisis",
       "de-dollarization", "BRICS".
   - Applies a non-linear power (`gamma = 1.5`) to emphasize strong, clear
     signals and down-weight ambiguous ones.

5. It multiplies **recency weight × impact weight** to get an **effective
   weight** for each article.

All of this produces:

- A list of `SentimentScores` (one per article).
- A list of corresponding **weights** that reflect both freshness and macro impact.

### 3. Index calculation (`processing/index_calc.py`)

Given all article-level scores and weights, it:

1. Computes a **weighted average** of positive, negative, and neutral
   probabilities across all documents.
2. Defines a net sentiment **NW** as:

   ```
   NW_raw = avg_positive - avg_negative   # range [-1, 1]
   NW = clip(SENSITIVITY * NW_raw, -1, 1)
   ```

   where `SENSITIVITY` is currently `2.2`, which makes the index move a bit more
   when the model is confident.

3. Normalizes `NW` from [-1, 1] to [0, 100]:

   ```
   nw_norm = (NW + 1.0) * 50.0
   gsi = nw_norm
   ```

4. Maps the resulting `gsi` value into a **regime label**:

   - `< 25`  → "Extremely Bearish"
   - `< 45`  → "Bearish"
   - `< 55`  → "Neutral"
   - `< 75`  → "Bullish"
   - `>= 75` → "Extremely Bullish"

The final output is packaged as `IndexComponents` with:

- `nw`      – net sentiment in [-1, 1]
- `nw_norm` – normalized to [0, 100]
- `gsi`     – alias for `nw_norm`
- `classification` – the regime label (Extremely Bearish / Bearish / Neutral / Bullish / Extremely Bullish)

### 4. Result packaging and dashboard (`processing/sentiment.py` & `processing/dashboard.py`)

- `processing/sentiment.save_results()` writes a detailed JSON structure to
  `sentiment_results.json`, including:
  - Timestamp of the run.
  - Per-article sentiment scores and source metadata.
  - Index components (`nw`, `nw_norm`, `gsi`, `classification`).
- It also writes a small `gsi_value.json` that just holds the current index and
  classification (handy for lightweight integrations).
- `processing/dashboard.generate_dashboard()`:
  - Reads `sentiment_results.json`.
  - Generates `dashboard.html` with an embedded `<canvas>` and custom drawing
    code (no backend server required).
- Draws:
    - The background bands (Extremely Bearish → Extremely Bullish).
    - A needle at the current GSI value.
    - Numeric value and regime text.
    - Last-updated time in local format.

---

## Why there were both `news.json` and `news_new.json`

Historically, the script wrote:

- `news.json` – the **full merged history** of all articles seen so far.
- `news_new.json` – only the **newly discovered** articles from the most recent
  fetch.

The rest of the pipeline (sentiment and index) only ever used `news.json`.
`news_new.json` was just a helper file and not actually needed.

To avoid confusion, the code has been simplified so that **only `news.json`
matters**. `news_new.json` is no longer written or used; you can delete it if
present.

---

## Rate limits, costs, and practical tips

- Free NewsAPI accounts are **limited** in how many requests and how much
  history you can fetch.
- Use `python run.py sentiment news` sparingly if you are on a free plan.
- If you hit HTTP errors from NewsAPI (e.g., 426 Upgrade Required), reduce the
  number of pages or the fetch frequency.
- If you want to backfill historical data, look at
  `scraping/newsapi.backfill_last_days()` and
  `scraping/newsapi.run_backfill_cli()` — they can walk backwards in time in
  windows, but may exceed free-tier limits.

---

## Troubleshooting

**1. `RuntimeError: NEWSAPI_KEY is not set in .env`**

- Ensure the `.env` file in the project root exists (it is included in the repo).
- Open it and make sure `NEWSAPI_KEY` is set to your real key, not the placeholder:

  ```ini
  NEWSAPI_KEY=abcdef1234567890
  ```

- Restart your terminal or re-activate your virtualenv so environment variables
  are picked up.

**2. `requests.exceptions.HTTPError` from NewsAPI**

- Check the printed message; common causes:
  - Invalid API key.
  - Exceeded plan quota.
  - Free tier limitations (too many pages or date range too wide).
- Try reducing `max_pages` and/or adjusting the date window.

**3. Model download is slow or fails**

- The first time you run FinBERT, `transformers` will download the model.
- Ensure you have a stable internet connection.
- If the process is killed due to memory, try running on a machine with more
  RAM or swap.

**4. Dashboard shows Neutral all the time**

- If `news.json` contains mostly non-macro / non-gold content or very
  short/ambiguous headlines, FinBERT may lean neutral.
- Try adjusting the keyword filters in `scraping/newsapi.py` to be stricter.
- You can also tweak the **recency** and **impact** weighting functions in
  `processing/sentiment.py` and `processing/index_calc.py` to make the index
  more or less sensitive.

---

## Extending the project

Ideas for future improvements:

- Add **Twitter / X** sentiment (requires separate API and keys; currently not
  used to keep setup simple).
- Store historical **GSI time series** (e.g., one value per day) and plot it as
  a chart under the gauge.
- Experiment with other domain models or fine-tuned gold-specific models.
- Deploy the dashboard as a small web app instead of a static file.

For now, the core path is:

1. Open `.env` and set `NEWSAPI_KEY` to your NewsAPI key.
2. `pip install -r requirements.txt`.
3. `python run.py sentiment update`.
4. Open `dashboard.html`.

That’s all you need to get from raw macro/gold headlines to a visual Gold

Sentiment Index.


