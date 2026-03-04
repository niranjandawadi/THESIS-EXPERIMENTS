from __future__ import annotations
import argparse
import yaml

from poverty_abm.experiments.runner import run_topology_sweep, run_seeding_experiment

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # For now: only topology sweep
    if cfg.get("experiment", {}).get("name") == "topology_sweep":
        run_topology_sweep(cfg)
    elif cfg.get("experiment", {}).get("name") == "seeding_aid":
        run_seeding_experiment(cfg)
    else:
        raise ValueError("Unknown experiment name.")

if __name__ == "__main__":
    main()