from __future__ import annotations
from typing import List, Dict, Any
import numpy as np
import networkx as nx

def select_targets(
    G: nx.Graph,
    wealth: np.ndarray,
    k: int,
    strategy: str,
    seed: int,
) -> List[int]:
    rng = np.random.default_rng(seed)
    n = G.number_of_nodes()
    nodes = np.array(list(G.nodes()))

    if k <= 0:
        return []
    k = min(k, n)

    if strategy == "random":
        return rng.choice(nodes, size=k, replace=False).tolist()

    if strategy == "poorest_first":
        # smallest wealth
        idx = np.argsort(wealth)[:k]
        return idx.tolist()

    if strategy == "degree":
        deg = np.array([G.degree(i) for i in range(n)], dtype=float)
        return np.argsort(-deg)[:k].tolist()

    if strategy in ("betweenness", "bridge_betweenness"):
        bc = nx.betweenness_centrality(G, normalized=True)
        score = np.array([bc[i] for i in range(n)], dtype=float)

        if strategy == "betweenness":
            return np.argsort(-score)[:k].tolist()

        # bridge_betweenness: emphasize nodes that connect communities.
        # Approx: high betweenness + diverse neighbor communities
        comm = nx.community.louvain_communities(G, seed=seed)
        comm_id = np.zeros(n, dtype=int)
        for cid, group in enumerate(comm):
            for v in group:
                comm_id[v] = cid

        diversity = np.zeros(n, dtype=float)
        for v in range(n):
            neigh = list(G.neighbors(v))
            if not neigh:
                diversity[v] = 0.0
            else:
                diversity[v] = len(set(comm_id[u] for u in neigh)) / max(1, len(comm))

        bridge_score = score * (1.0 + diversity)
        return np.argsort(-bridge_score)[:k].tolist()

    if strategy == "pagerank":
        pr = nx.pagerank(G, alpha=0.85)
        score = np.array([pr[i] for i in range(n)], dtype=float)
        return np.argsort(-score)[:k].tolist()

    if strategy in ("community_spread", "community_clustered"):
        # community detection
        comm = nx.community.louvain_communities(G, seed=seed)
        comm = [sorted(list(c)) for c in comm]
        if len(comm) == 0:
            return rng.choice(nodes, size=k, replace=False).tolist()

        if strategy == "community_spread":
            # pick ~one per community, cycling, with poorest in each community
            targets = []
            # sort each community by wealth (poorest first)
            comm_sorted = [sorted(c, key=lambda v: wealth[v]) for c in comm]
            i = 0
            while len(targets) < k:
                c = comm_sorted[i % len(comm_sorted)]
                # pick first available not yet chosen
                for v in c:
                    if v not in targets:
                        targets.append(v)
                        break
                i += 1
                if i > 10_000:
                    break
            if len(targets) < k:
                remaining = [v for v in range(n) if v not in targets]
                extra = rng.choice(remaining, size=min(k - len(targets), len(remaining)), replace=False).tolist()
                targets.extend(extra)
            return targets[:k]

        if strategy == "community_clustered":
            # choose one community and pick k poorest from that community
            cid = rng.integers(0, len(comm))
            c = comm[cid]
            c_sorted = sorted(c, key=lambda v: wealth[v])
            if len(c_sorted) >= k:
                return c_sorted[:k]
            # if community too small, fill from next poorest overall
            targets = c_sorted
            remaining = [v for v in np.argsort(wealth) if v not in targets]
            targets.extend(remaining[: (k - len(targets))])
            return targets[:k]

    raise ValueError(f"Unknown strategy: {strategy}")