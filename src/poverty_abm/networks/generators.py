from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List
import networkx as nx
import numpy as np

def make_graph(graph_type: str, n: int, seed: int, params: Dict[str, Any]) -> nx.Graph:
    rng = np.random.default_rng(seed)

    if graph_type == "er":
        p = float(params["p"])
        G = nx.erdos_renyi_graph(n=n, p=p, seed=seed)
    elif graph_type == "ws":
        k = int(params["k"])
        p_rewire = float(params["p_rewire"])
        G = nx.watts_strogatz_graph(n=n, k=k, p=p_rewire, seed=seed)
    elif graph_type == "ba":
        m = int(params["m"])
        G = nx.barabasi_albert_graph(n=n, m=m, seed=seed)
    elif graph_type == "sbm":
        sizes: List[int] = list(params["sizes"])
        if sum(sizes) != n:
            raise ValueError(f"SBM sizes must sum to n={n}. Got sum={sum(sizes)}")
        p_in = float(params["p_in"])
        p_out = float(params["p_out"])
        # block probability matrix
        B = [[p_in if i == j else p_out for j in range(len(sizes))] for i in range(len(sizes))]
        G = nx.stochastic_block_model(sizes, B, seed=seed)
    else:
        raise ValueError(f"Unknown graph_type: {graph_type}")

    # Ensure connectedness for fair comparison (optional but recommended)
    if not nx.is_connected(G):
        # Take largest connected component to avoid weird isolated nodes
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
        # If component smaller than n, relabel and warn via exception or handle:
        if G.number_of_nodes() < n:
            # For now, relabel nodes 0..n_cc-1 and continue; runner will track actual n.
            G = nx.convert_node_labels_to_integers(G)

    return G
