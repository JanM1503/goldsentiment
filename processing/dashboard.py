from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SENTIMENT_RESULTS_PATH = PROJECT_ROOT / "docs" / "sentiment_results.json"
DASHBOARD_HTML_PATH = PROJECT_ROOT / "docs" / "index.html"


def _load_current_result() -> Dict[str, Any] | None:
    """Load the latest sentiment_results.json produced by analysis.

    This file contains the current GSI, classification, and timestamp.
    """

    if not SENTIMENT_RESULTS_PATH.exists():
        return None
    try:
        return json.loads(SENTIMENT_RESULTS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


def generate_dashboard() -> None:
    """Generate a dashboard that loads the latest GSI from gsi_value.json.

    This HTML is static and can be hosted on GitHub Pages. The heavy work
    (NewsAPI + FinBERT + index calculation) still runs in Python and stores
    the current value in ``gsi_value.json``. When the page is opened, a small
    JavaScript loop fetches that JSON file and updates the gauge.
    """

    html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Gold Sentiment Index Dashboard</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; padding: 30px; background: #f3f4f6; }
    h1 { margin-bottom: 0.4rem; }
    .subtitle { color: #4b5563; margin-bottom: 2rem; }
    .card { max-width: 640px; margin: 0 auto; box-shadow: 0 8px 20px rgba(15,23,42,0.12); padding: 30px 30px 40px; border-radius: 18px; background: #ffffff; }
    .gauge-wrapper {
      position: relative;
      width: 100%;
      max-width: 540px;
      margin: 0 auto;
    }
    .gauge-value { font-size: 48px; font-weight: 700; text-align: center; margin-top: 16px; }
    .gauge-label { text-align: center; font-weight: 600; letter-spacing: 0.08em; color: #374151; text-transform: uppercase; margin-top: 4px; }
    .updated { font-size: 12px; color: #6b7280; margin-top: 10px; text-align: center; }
    .bands { display: flex; justify-content: space-between; font-size: 11px; text-transform: uppercase; color: #4b5563; margin-top: 8px; padding: 0 8px; }
    .bands span { flex: 1; text-align: center; }
    .bands span:first-child { text-align: left; }
    .bands span:last-child { text-align: right; }
    canvas { max-width: 540px; display: block; margin: 0 auto; }
  </style>
</head>
<body>
  <h1>Gold Sentiment Index (GSI)</h1>
  <p class="subtitle">News-driven gold sentiment, scaled 0–100 (Extremely Bearish → Extremely Bullish).</p>

  <div class="card">
    <div class="gauge-wrapper">
      <canvas id="gaugeChart" width="540" height="280"></canvas>
    </div>
    <div class="bands">
      <span>Extreme Bearish<br/>0–25</span>
      <span>Bearish<br/>25–45</span>
      <span>Neutral<br/>45–55</span>
      <span>Bullish<br/>55–75</span>
      <span>Extreme Bullish<br/>75–100</span>
    </div>
    <div class="gauge-value" id="gaugeValue">--</div>
    <div class="gauge-label" id="gaugeLabel">Loading…</div>
    <div class="updated" id="updatedText">Fetching latest data…</div>
  </div>

  <script>
    (function () {
      const canvas = document.getElementById('gaugeChart');
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      const W = canvas.width;
      const H = canvas.height;
      const cx = W / 2;
      const cy = H * 0.9;          // center near bottom like CNN gauge
      const outerR = Math.min(W * 0.9, H * 1.8) / 2;
      const innerR = outerR * 0.6;

      const valueEl = document.getElementById('gaugeValue');
      const labelEl = document.getElementById('gaugeLabel');
      const updatedEl = document.getElementById('updatedText');

      function thetaFor(value) {
        // 0 → left (π), 50 → top (π/2), 100 → right (0)
        return Math.PI * (1 - value / 100);
      }

      function drawBand(startVal, endVal, color) {
        const start = thetaFor(startVal);
        const end = thetaFor(endVal);

        ctx.beginPath();
        // Outer arc: left → right along top (counterclockwise)
        ctx.arc(cx, cy, outerR, start, end, true);
        // Inner arc back: right → left along inner radius (clockwise)
        ctx.arc(cx, cy, innerR, end, start, false);
        ctx.closePath();
        ctx.fillStyle = color;
        ctx.fill();
      }

      function drawGauge(value) {
        ctx.clearRect(0, 0, W, H);

        // Background bands: Extreme Bearish, Bearish, Neutral, Bullish, Extreme Bullish
        drawBand(0, 25,  '#ff0000');  // red
        drawBand(25, 45, '#fee2e2');  // light red
        drawBand(45, 55, '#e5e7eb');  // gray
        drawBand(55, 75, '#bbf7d0');  // light green
        drawBand(75, 100,'#22c55e');  // green

        // Center white circle for numeric value
        ctx.save();
        ctx.beginPath();
        ctx.arc(cx, cy, innerR * 0.9, 0, Math.PI * 2);
        ctx.fillStyle = '#ffffff';
        ctx.shadowColor = 'rgba(0,0,0,0.08)';
        ctx.shadowBlur = 12;
        ctx.fill();
        ctx.restore();

        // Needle
        const th = thetaFor(value);
        const nx = cx + Math.cos(th) * (outerR * 0.9);
        const ny = cy - Math.sin(th) * (outerR * 0.9);

        ctx.save();
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(nx, ny);
        ctx.lineWidth = 4;
        ctx.strokeStyle = '#111827';
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(cx, cy, 6, 0, Math.PI * 2);
        ctx.fillStyle = '#111827';
        ctx.fill();
        ctx.restore();
      }

      function classifyGsi(gsi) {
        if (gsi < 25) return 'Extremely Bearish';
        if (gsi < 45) return 'Bearish';
        if (gsi < 55) return 'Neutral';
        if (gsi < 75) return 'Bullish';
        return 'Extremely Bullish';
      }

      function formatTimestamp(tsRaw) {
        if (!tsRaw) return '';
        try {
          const d = new Date(tsRaw);
          return d.toLocaleString('en-US', {
            year: 'numeric', month: 'short', day: 'numeric',
            hour: 'numeric', minute: '2-digit',
            timeZoneName: 'short'
          });
        } catch (e) {
          return tsRaw;
        }
      }

      function applyPayload(payload) {
        const gsi = Number(payload && payload.gsi != null ? payload.gsi : 50);
        const classification = payload && payload.classification ? payload.classification : classifyGsi(gsi);
        const ts = payload && payload.timestamp ? payload.timestamp : '';

        drawGauge(gsi);
        if (valueEl) valueEl.textContent = gsi.toFixed(0);
        if (labelEl) labelEl.textContent = classification;
        if (updatedEl) {
          const formatted = formatTimestamp(ts);
          updatedEl.textContent = 'Last updated: ' + (formatted || 'n/a');
        }
      }

      async function fetchAndUpdate() {
        try {
          if (updatedEl) updatedEl.textContent = 'Fetching latest data…';
          const resp = await fetch('gsi_value.json?cache=' + Date.now());
          if (!resp.ok) throw new Error('HTTP ' + resp.status);
          const data = await resp.json();
          applyPayload(data);
        } catch (err) {
          console.error('Failed to load GSI', err);
          if (updatedEl) updatedEl.textContent = 'Failed to load latest data.';
        }
      }

      // Initial load and refresh loop (e.g. every 5 minutes)
      fetchAndUpdate();
      setInterval(fetchAndUpdate, 5 * 60 * 1000);
    })();
  </script>
</body>
</html>
"""

    DASHBOARD_HTML_PATH.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    generate_dashboard()



