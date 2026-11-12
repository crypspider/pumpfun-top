#!/usr/bin/env python3
"""Forecaster for make.fun bounty pumpfun-top5-forecast-01.

Reads the house's 10-minute market-cap snapshots from $MAKEFUN_INPUT_DIR
(default /input): task.json (places, size-bin edges, output name, answer
reads) and history.ndjson (one mark per line, rows ranked by market cap).
Writes one forecast to $MAKEFUN_OUTPUT_DIR (default /output)/forecast.json:

  {"probabilities": [{"mint": ..., "place": k, "p": ...}, ...],
   "top5": [mint, ...],
   "sizes": [{"place": k, "bin": b, "p": ...}, ...]}

Method (deterministic, stdlib only, no network):
  * For every coin in the history, estimate its current log10 market cap
    (in SOL), its trend (least-squares slope of log cap over time plus the
    recent change), and its volatility (spread of per-mark log changes).
  * Project each coin to the middle of the answer window with a damped
    drift (mean reversion: only 38% of the extrapolated trend is kept,
    capped), so fast risers gain mass over the naive "last order holds"
    baseline without extrapolating a 10-minute pump over 24 hours.
  * Simulate the answer reads with a fixed-seed RNG: each candidate's
    future log cap is drawn from a normal around its projection with a
    per-coin sigma that grows with the horizon and the coin's volatility.
    Place probabilities and per-place size-bin probabilities are the
    simulated frequencies, blended slightly with analytic priors.
  * Each place deliberately leaves part of its probability unnamed: coins
    born after the cut (and coins outside the candidate set) can take a
    place, and the scorer shares the unnamed part among them.
Two runs on the same input write byte-identical output (fixed seed, fixed
iteration order, probabilities floored to 6 decimals like the baseline).
"""

import json
import math
import os
import random
from bisect import bisect_right

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

SEED = 20261112
DRAWS = 20000
MAX_CANDIDATES = 60
DRIFT_KEEP = 0.38          # fraction of extrapolated trend kept
DRIFT_MIN, DRIFT_MAX = -0.90, 0.65   # log10 caps on the projected drift
MIN_NAMED_P = 0.002        # drop named place rows below this (after scaling)
MICRO = 1_000_000


def parse_time(s):
    """ISO-8601 'Z' timestamp -> epoch seconds (float), or None."""
    if not s:
        return None
    try:
        from datetime import datetime, timezone
        return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S.%fZ").replace(
            tzinfo=timezone.utc).timestamp()
    except Exception:
        try:
            from datetime import datetime, timezone
            return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(
                tzinfo=timezone.utc).timestamp()
        except Exception:
            return None


def logcap(lamports):
    """log10 of a market cap in SOL, as the scorer takes it."""
    return math.log10(int(lamports)) - 9.0


def norm_cdf(x):
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def main():
    with open(os.path.join(INPUT_DIR, "task.json"), "r", encoding="utf-8") as f:
        task = json.load(f)
    places = int(task.get("places", 5))
    edges = [float(e) for e in task["size"]["edges"]]
    out_name = task.get("output", "forecast.json")

    marks = []  # (time_seconds, {mint: (logcap, rank)})
    hist_mints = set()
    with open(os.path.join(INPUT_DIR, "history.ndjson"), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                m = json.loads(line)
            except Exception:
                continue
            if m.get("ok") is not True or not isinstance(m.get("rows"), list):
                continue
            t = parse_time(m.get("at"))
            rows = {}
            for r in m["rows"]:
                try:
                    rows[r["mint"]] = (logcap(r["market_cap_lamports"]), int(r["rank"]))
                    hist_mints.add(r["mint"])
                except Exception:
                    continue
            if rows:
                marks.append((t if t is not None else float(len(marks)), rows))
    marks.sort(key=lambda x: x[0])

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, out_name)

    if not marks:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"probabilities": [], "top5": [], "sizes": []}) + "\n")
        return

    # Horizon: history end -> middle of the answer reads (hours).
    t_last = marks[-1][0]
    read_times = [parse_time(s) for s in task.get("reads", [])]
    read_times = [t for t in read_times if t is not None]
    if read_times:
        horizon_h = max(0.0, (sum(read_times) / len(read_times) - t_last) / 3600.0)
    else:
        horizon_h = 6.0
    horizon_h = min(horizon_h, 24.0)

    # Per-coin series: list of (t_hours_from_first, logcap).
    series = {}
    for t, rows in marks:
        th = (t - marks[0][0]) / 3600.0
        for mint, (lc, _rank) in rows.items():
            series.setdefault(mint, []).append((th, lc))
    n_marks = len(marks)
    last_rows = marks[-1][1]
    span_h = max((marks[-1][0] - marks[0][0]) / 3600.0, 1e-9)

    stats = {}
    for mint, pts in series.items():
        pts.sort()
        last_t, last_lc = pts[-1]
        marks_since_seen = 0
        for t, rows in reversed(marks):
            if mint in rows:
                break
            marks_since_seen += 1
        # Current level: if absent from the last mark, discount the last
        # level seen (it fell out of the listed rows).
        if mint in last_rows:
            current = last_rows[mint][0]
        else:
            current = last_lc - 0.25 * marks_since_seen
        # Trend: least-squares slope (log10 per hour) over the recent window.
        win = pts[-18:]
        slope = 0.0
        if len(win) >= 2:
            mt = sum(p[0] for p in win) / len(win)
            mv = sum(p[1] for p in win) / len(win)
            num = sum((p[0] - mt) * (p[1] - mv) for p in win)
            den = sum((p[0] - mt) ** 2 for p in win)
            if den > 1e-12:
                slope = num / den
        recent = 0.0
        if len(pts) >= 2:
            recent = pts[-1][1] - pts[max(0, len(pts) - 4)][1]
        # Volatility: stdev of per-step log changes, scaled to the horizon.
        vol = 0.0
        if len(pts) >= 3:
            diffs = [pts[i + 1][1] - pts[i][1] for i in range(len(pts) - 1)]
            md = sum(diffs) / len(diffs)
            step_sd = math.sqrt(sum((d - md) ** 2 for d in diffs) / len(diffs))
            steps = max(1.0, (pts[-1][0] - pts[0][0]) * 6.0)  # 10-min steps
            per_hour_sd = step_sd * math.sqrt(6.0)
            vol = min(0.55, per_hour_sd * math.sqrt(max(horizon_h, 0.25)) / max(1.0, math.sqrt(steps / 6.0)) * 0.35)
        drift = DRIFT_KEEP * slope * horizon_h + 0.20 * recent * min(1.0, horizon_h / 4.0)
        drift = max(DRIFT_MIN, min(DRIFT_MAX, drift))
        proj = current + drift
        sigma = 0.30 + 0.085 * math.sqrt(horizon_h) + vol
        if mint not in last_rows:
            sigma += 0.20
        sigma = max(0.35, min(1.10, sigma))
        stats[mint] = (proj, sigma, current, slope)

    # Candidates: best projections, plus the fastest risers just below them.
    by_proj = sorted(stats.items(), key=lambda kv: (-kv[1][0], kv[0]))
    cand = [m for m, _ in by_proj[:48]]
    fifth = by_proj[min(places, len(by_proj)) - 1][1][0] if by_proj else 0.0
    risers = sorted(
        ((m, s) for m, s in stats.items() if m not in cand and s[0] > fifth - 1.2),
        key=lambda kv: (-kv[1][3], kv[0]))
    cand += [m for m, _ in risers[:12]]
    cand = cand[:MAX_CANDIDATES]
    top5 = [m for m, _ in by_proj[:places]]

    # Seeded simulation of the answer reads.
    rng = random.Random(SEED)
    params = [(m, stats[m][0], stats[m][1]) for m in cand]
    place_counts = {m: [0] * places for m in cand}
    size_counts = [[0] * (len(edges) + 1) for _ in range(places)]
    for _ in range(DRAWS):
        draw = sorted(((rng.gauss(mu, sd), m) for m, mu, sd in params),
                      key=lambda x: (-x[0], x[1]))
        for k in range(min(places, len(draw))):
            val, m = draw[k]
            place_counts[m][k] += 1
            size_counts[k][bisect_right(edges, val)] += 1

    probabilities = []
    for k in range(places):
        # Unnamed share for this place: newcomers and non-candidates.
        unnamed = min(0.50, 0.10 + 0.025 * horizon_h + 0.02 * k)
        scale = 1.0 - unnamed
        rows = []
        for m in cand:
            c = place_counts[m][k]
            if c == 0:
                continue
            p = (c / DRAWS) * scale
            u = math.floor(p * MICRO)
            if u / MICRO >= MIN_NAMED_P:
                rows.append((m, u / MICRO))
        rows.sort(key=lambda x: (-x[1], x[0]))
        for m, p in rows[:100]:
            probabilities.append({"mint": m, "place": k + 1, "p": p})

    sizes = []
    for k in range(places):
        counts = size_counts[k]
        total = sum(counts)
        if total == 0:
            continue
        # Blend the simulated frequencies with a normal prior centred on the
        # k-th best projection, so no plausible bin is left at the floor.
        centre = sorted((stats[m][0] for m in cand), reverse=True)[k] if len(cand) > k else 0.0
        sd_prior = 0.55 + 0.06 * math.sqrt(horizon_h)
        for b in range(len(edges) + 1):
            lo = -math.inf if b == 0 else edges[b - 1]
            hi = math.inf if b == len(edges) else edges[b]
            prior = norm_cdf((hi - centre) / sd_prior) - norm_cdf((lo - centre) / sd_prior)
            p = 0.85 * (counts[b] / total) + 0.15 * prior
            u = math.floor(p * MICRO)
            if u > 0:
                sizes.append({"place": k + 1, "bin": b, "p": u / MICRO})

    forecast = {"probabilities": probabilities, "top5": top5, "sizes": sizes}
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(forecast) + "\n")


if __name__ == "__main__":
    main()
