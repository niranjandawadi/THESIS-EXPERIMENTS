from __future__ import annotations
from typing import Dict, Any, List
import os, json
import numpy as np
import pandas as pd

from poverty_abm.networks.generators import make_graph
from poverty_abm.model.abm import run_simulation
from poverty_abm.eval.measures import compute_all
from poverty_abm.experiments.seeds import select_targets
from poverty_abm.eval.spillovers import compute_spillovers

def run_topology_sweep(cfg: Dict[str, Any]) -> None:
    exp_name = cfg["experiment"]["name"]
    out_root = cfg["experiment"]["output_dir"]
    os.makedirs(out_root, exist_ok=True)

    n_agents = int(cfg["sim"]["n_agents"])
    n_runs = int(cfg["sim"]["n_runs"])
    base_seed = int(cfg["sim"]["seed"])

    cases: List[Dict[str, Any]] = cfg["topology_sweep"]["cases"]

    all_rows = []
    ts_rows = []  # time series (averaged later)

    for case_idx, case in enumerate(cases):
        graph_type = case["graph_type"]
        params = case["params"]

        for run_id in range(n_runs):
            seed = base_seed + 10_000 * case_idx + run_id

            G = make_graph(graph_type=graph_type, n=n_agents, seed=seed, params=params)
            run = run_simulation(G, cfg, seed=seed)
            metrics = compute_all(run, cfg)

            # per-run summary (thesis tables)
            all_rows.append({
                "experiment": exp_name,
                "graph_type": graph_type,
                "params": json.dumps(params, sort_keys=True),
                "run_id": run_id,
                "n": run["n"],
                "escape_fraction": metrics["escape_fraction"],
                "escape_time_mean": metrics["escape_time_mean"],
                "escape_time_median": metrics["escape_time_median"],
                "gini_final": float(metrics["gini_ts"][-1]),
                "poverty_final": float(metrics["poverty_ts"][-1]),
                "spearman_rank_corr": metrics["spearman_rank_corr"],
                "mean_abs_rank_change": metrics["mean_abs_rank_change"],
            })

            # time series rows (for plots)
            for t, (gini_t, pov_t) in enumerate(zip(metrics["gini_ts"], metrics["poverty_ts"])):
                ts_rows.append({
                    "graph_type": graph_type,
                    "params": json.dumps(params, sort_keys=True),
                    "run_id": run_id,
                    "t": t,
                    "gini": float(gini_t),
                    "poverty_rate": float(pov_t),
                })

    df_runs = pd.DataFrame(all_rows)
    df_ts = pd.DataFrame(ts_rows)

    # output folder
    out_dir = os.path.join(out_root, exp_name)
    os.makedirs(out_dir, exist_ok=True)

    # save config snapshot
    with open(os.path.join(out_dir, "config_used.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    df_runs.to_csv(os.path.join(out_dir, "runs_summary.csv"), index=False)
    df_ts.to_csv(os.path.join(out_dir, "runs_timeseries.csv"), index=False)

    # aggregate time series (mean +/- std)
    agg = df_ts.groupby(["graph_type", "params", "t"]).agg(
        gini_mean=("gini", "mean"),
        gini_std=("gini", "std"),
        pov_mean=("poverty_rate", "mean"),
        pov_std=("poverty_rate", "std"),
    ).reset_index()
    agg.to_csv(os.path.join(out_dir, "timeseries_agg.csv"), index=False)

    # aggregate run summary (means)
    table = df_runs.groupby(["graph_type", "params"]).agg(
        escape_fraction_mean=("escape_fraction", "mean"),
        escape_time_mean=("escape_time_mean", "mean"),
        gini_final_mean=("gini_final", "mean"),
        poverty_final_mean=("poverty_final", "mean"),
        spearman_rank_corr_mean=("spearman_rank_corr", "mean"),
        mean_abs_rank_change_mean=("mean_abs_rank_change", "mean"),
        n_mean=("n", "mean"),
    ).reset_index()
    table.to_csv(os.path.join(out_dir, "summary_table.csv"), index=False)

    print(f"Saved results to: {out_dir}")

def run_seeding_experiment(cfg: Dict[str, Any]) -> None:
    exp_name = cfg["experiment"]["name"]
    out_root = cfg["experiment"]["output_dir"]
    os.makedirs(out_root, exist_ok=True)

    n_agents = int(cfg["sim"]["n_agents"])
    n_runs = int(cfg["sim"]["n_runs"])
    base_seed = int(cfg["sim"]["seed"])
    survival = float(cfg["economy"]["survival_threshold"])

    # fixed network for this experiment (per run we can keep same or regenerate per seed)
    net_cfg = cfg["network"]
    graph_type = net_cfg["graph_type"]
    graph_params = net_cfg["params"]

    strategies = cfg["seeding_experiment"]["strategies"]
    k = int(cfg["aid"]["k"])

    all_rows = []
    ts_rows = []

    for strat_idx, strategy in enumerate(strategies):
        for run_id in range(n_runs):
            seed = base_seed + 10_000 * strat_idx + run_id

            G = make_graph(graph_type=graph_type, n=n_agents, seed=seed, params=graph_params)

            # initial wealth (need it to pick poorest_first)
            # run_simulation initializes wealth internally, so we replicate init here:
            # easiest: run a "dry" init by calling run_simulation with 0 steps? (but we want clean)
            # We'll re-use the sim init logic by importing init_wealth
            from poverty_abm.model.abm import init_wealth
            rng = np.random.default_rng(seed)
            w0 = init_wealth(G.number_of_nodes(), cfg, rng)

            targets = select_targets(G, w0, k=k, strategy=strategy, seed=seed)

            # pass targets into cfg for this run (copy to avoid cross-run mutation)
            cfg_run = dict(cfg)
            cfg_run["_targets"] = targets

            run = run_simulation(G, cfg_run, seed=seed)
            metrics = compute_all(run, cfg_run)

            spill = compute_spillovers(G, run["wealth_history"][0], run["wealth_history"][-1], targets, survival)

            all_rows.append({
                "experiment": exp_name,
                "strategy": strategy,
                "run_id": run_id,
                "n": run["n"],
                "k": k,
                "budget_total": float(cfg["aid"]["budget_total"]),
                "escape_fraction": metrics["escape_fraction"],
                "escape_time_mean": metrics["escape_time_mean"],
                "gini_final": float(metrics["gini_ts"][-1]),
                "poverty_final": float(metrics["poverty_ts"][-1]),
                "spearman_rank_corr": metrics["spearman_rank_corr"],
                "mean_abs_rank_change": metrics["mean_abs_rank_change"],
                **spill,
            })

            for t, (gini_t, pov_t) in enumerate(zip(metrics["gini_ts"], metrics["poverty_ts"])):
                ts_rows.append({
                    "strategy": strategy,
                    "run_id": run_id,
                    "t": t,
                    "gini": float(gini_t),
                    "poverty_rate": float(pov_t),
                })

    df_runs = pd.DataFrame(all_rows)
    df_ts = pd.DataFrame(ts_rows)

    out_dir = os.path.join(out_root, exp_name)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "config_used.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    df_runs.to_csv(os.path.join(out_dir, "runs_summary.csv"), index=False)
    df_ts.to_csv(os.path.join(out_dir, "runs_timeseries.csv"), index=False)

    agg = df_ts.groupby(["strategy", "t"]).agg(
        gini_mean=("gini", "mean"),
        gini_std=("gini", "std"),
        pov_mean=("poverty_rate", "mean"),
        pov_std=("poverty_rate", "std"),
    ).reset_index()
    agg.to_csv(os.path.join(out_dir, "timeseries_agg.csv"), index=False)

    table = df_runs.groupby(["strategy"]).agg(
        poverty_final_mean=("poverty_final", "mean"),
        gini_final_mean=("gini_final", "mean"),
        escape_fraction_mean=("escape_fraction", "mean"),
        escape_time_mean=("escape_time_mean", "mean"),
        treated_mean_dW_mean=("treated_mean_dW", "mean"),
        non_treated_mean_dW_mean=("non_treated_mean_dW", "mean"),
        neighbor_mean_dW_mean=("neighbor_mean_dW", "mean"),
        non_treated_poverty_final_mean=("non_treated_poverty_final", "mean"),
        neighbor_poverty_final_mean=("neighbor_poverty_final", "mean"),
        neighbor_count_mean=("neighbor_count", "mean"),
    ).reset_index()
    table.to_csv(os.path.join(out_dir, "summary_table.csv"), index=False)

    print(f"Saved results to: {out_dir}")