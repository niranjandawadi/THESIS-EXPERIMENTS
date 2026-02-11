from __future__ import annotations
from typing import Dict, Any
import numpy as np

def gini(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if np.allclose(x, 0):
        return 0.0
    x = np.clip(x, 0, None)
    xs = np.sort(x)
    n = xs.size
    cum = np.cumsum(xs)
    # Gini = (n+1 - 2 * sum_i (cum_i / cum_n)) / n
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)

def poverty_rate(x: np.ndarray, threshold: float) -> float:
    return float(np.mean(x < threshold))

def time_to_escape(wealth_history: np.ndarray, threshold: float) -> Dict[str, Any]:
    """
    For agents who start below threshold, compute first time they reach >= threshold.
    wealth_history: (T, n)
    """
    T, n = wealth_history.shape
    start_below = wealth_history[0] < threshold

    escape_times = np.full(n, fill_value=np.nan, dtype=float)
    for i in range(n):
        if not start_below[i]:
            continue
        hits = np.where(wealth_history[:, i] >= threshold)[0]
        if hits.size > 0:
            escape_times[i] = float(hits[0])  # index in recorded steps

    eligible = np.where(start_below)[0]
    escaped = np.isfinite(escape_times[eligible])

    return {
        "n_start_below": int(eligible.size),
        "escape_fraction": float(np.mean(escaped)) if eligible.size else np.nan,
        "escape_time_mean": float(np.nanmean(escape_times[eligible])) if np.any(escaped) else np.nan,
        "escape_time_median": float(np.nanmedian(escape_times[eligible])) if np.any(escaped) else np.nan,
    }

def rank_mobility(w0: np.ndarray, wT: np.ndarray) -> Dict[str, Any]:
    """
    Mobility via rank correlation and mean absolute rank change.
    """
    n = w0.size
    r0 = np.argsort(np.argsort(w0))  # 0..n-1 ranks
    rT = np.argsort(np.argsort(wT))

    # Spearman correlation (computed manually)
    r0c = r0 - r0.mean()
    rTc = rT - rT.mean()
    denom = (np.linalg.norm(r0c) * np.linalg.norm(rTc))
    spearman = float(np.dot(r0c, rTc) / denom) if denom > 0 else np.nan

    mean_abs_rank_change = float(np.mean(np.abs(rT - r0)))
    return {
        "spearman_rank_corr": spearman,
        "mean_abs_rank_change": mean_abs_rank_change,
    }

def compute_all(run: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    wh = run["wealth_history"]
    threshold = float(cfg["economy"]["survival_threshold"])

    gini_ts = np.array([gini(wh[t]) for t in range(wh.shape[0])], dtype=float)
    pov_ts = np.array([poverty_rate(wh[t], threshold) for t in range(wh.shape[0])], dtype=float)

    escape = time_to_escape(wh, threshold)
    mobility = rank_mobility(wh[0], wh[-1])

    return {
        "gini_ts": gini_ts,
        "poverty_ts": pov_ts,
        **escape,
        **mobility,
    }