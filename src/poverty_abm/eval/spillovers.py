from __future__ import annotations
from typing import Dict, Any, List
import numpy as np
import networkx as nx

def compute_spillovers(G: nx.Graph, wealth0: np.ndarray, wealthT: np.ndarray, targets: List[int], survival: float) -> Dict[str, Any]:
    n = len(wealth0)
    treated = np.zeros(n, dtype=bool)
    treated[targets] = True

    non_treated = ~treated

    # neighbors of treated, excluding treated
    neigh = set()
    for v in targets:
        for u in G.neighbors(v):
            if not treated[u]:
                neigh.add(u)
    neigh = np.array(sorted(list(neigh)), dtype=int)

    dW = wealthT - wealth0

    out = {
        "treated_mean_dW": float(np.mean(dW[treated])) if treated.any() else np.nan,
        "non_treated_mean_dW": float(np.mean(dW[non_treated])) if non_treated.any() else np.nan,
        "neighbor_mean_dW": float(np.mean(dW[neigh])) if neigh.size > 0 else np.nan,
        "non_treated_poverty_final": float(np.mean(wealthT[non_treated] < survival)) if non_treated.any() else np.nan,
        "treated_poverty_final": float(np.mean(wealthT[treated] < survival)) if treated.any() else np.nan,
        "neighbor_poverty_final": float(np.mean(wealthT[neigh] < survival)) if neigh.size > 0 else np.nan,
        "neighbor_count": int(neigh.size),
    }
    return out