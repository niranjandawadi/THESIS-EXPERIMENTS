from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np
import networkx as nx

@dataclass
class SimState:
    wealth: np.ndarray  # shape (n,)

def init_wealth(n: int, cfg: Dict[str, Any], rng: np.random.Generator) -> np.ndarray:
    init_cfg = cfg["economy"]["init_wealth"]
    total = float(init_cfg["total"])
    method = init_cfg["method"]

    if method == "equal":
        w = np.full(n, total / n, dtype=float)
    elif method == "lognormal":
        # fixed-shape distribution, normalized to same total
        mu = float(init_cfg.get("mu", 0.0))
        sigma = float(init_cfg.get("sigma", 1.0))
        raw = rng.lognormal(mean=mu, sigma=sigma, size=n)
        w = raw / raw.sum() * total
    else:
        raise ValueError(f"Unknown init wealth method: {method}")
    return w

def step_exchange(G: nx.Graph, state: SimState, cfg: Dict[str, Any], rng: np.random.Generator) -> None:
    transfer = float(cfg["exchange"]["step"]["transfer_amount"])
    direction = cfg["exchange"]["step"]["direction"]
    floor = float(cfg["exchange"]["floor"])

    edges = list(G.edges())
    if not edges:
        return

    u, v = edges[rng.integers(0, len(edges))]

    # pick direction
    if direction == "random":
        if rng.random() < 0.5:
            src, dst = u, v
        else:
            src, dst = v, u
    elif direction == "poor_to_rich":
        src, dst = (u, v) if state.wealth[u] < state.wealth[v] else (v, u)
    elif direction == "rich_to_poor":
        src, dst = (u, v) if state.wealth[u] > state.wealth[v] else (v, u)
    else:
        raise ValueError(f"Unknown direction: {direction}")

    # apply transfer with floor constraint
    amt = min(transfer, max(0.0, state.wealth[src] - floor))
    state.wealth[src] -= amt
    state.wealth[dst] += amt

def run_simulation(G: nx.Graph, cfg: Dict[str, Any], seed: int) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = G.number_of_nodes()

    wealth = init_wealth(n, cfg, rng)
    state = SimState(wealth=wealth)

    n_steps = int(cfg["sim"]["n_steps"])
    record_every = int(cfg["sim"].get("record_every", 1))

    # store time series metrics later (computed in measures)
    wealth_history = []
    for t in range(n_steps):
        step_exchange(G, state, cfg, rng)
        if (t % record_every) == 0:
            wealth_history.append(state.wealth.copy())

    return {
        "wealth_history": np.stack(wealth_history, axis=0),  # (T, n)
        "final_wealth": state.wealth.copy(),
        "n": n,
        "seed": seed,
    }
