# Project Structure

```
poverty-net-abm/                     ← your whole project folder (repo)
│
├─ README.md                         ← what the project is + how to run
├─ pyproject.toml  (or requirements.txt) ← dependencies
├─ .gitignore                        ← files Git should ignore (results, caches, etc.)
│
├─ src/                              ← ALL Python code lives here
│  └─ poverty_abm/                   ← your Python package (importable)
│     ├─ __init__.py                 ← makes it a package
│     │
│     ├─ model/                      ← “the simulator” (core logic)
│     │  ├─ abm.py                   ← main simulation loop: run steps, update state
│     │  ├─ agents.py                ← agent data + behavior rules
│     │  ├─ exchange.py              ← token exchange / transactions rules
│     │  ├─ interventions.py         ← policies: cash aid, loans, add edges, frictions
│     │  └─ shocks.py                ← optional: random shocks, disasters, job loss
│     │
│     ├─ networks/                   ← how you build + analyze networks
│     │  ├─ generators.py            ← create graphs (ER/WS/BA/SBM)
│     │  └─ metrics.py               ← centrality, bridges, community detection
│     │
│     ├─ experiments/                ← “how you run experiments”
│     │  ├─ runner.py                ← run ONE experiment from ONE config file
│     │  ├─ sweep.py                 ← run MANY configs (grid search / random sweep)
│     │  └─ seeds.py                 ← target selection: random/poorest/central/bridge
│     │
│     ├─ eval/                       ← “how you measure outcomes”
│     │  ├─ measures.py              ← compute poverty rate, Gini, mobility, dwell time
│     │  └─ aggregations.py          ← average across runs/seeds; summary tables
│     │
│     └─ utils/                      ← boring but important helpers
│        ├─ config.py                ← load/validate YAML config
│        ├─ rng.py                   ← consistent random seeding
│        └─ io.py                    ← save CSV/JSON/plots/logs to results/
│
├─ configs/                          ← “input recipes” for experiments (YAML files)
│  ├─ base.yaml
│  ├─ exp_topology.yaml
│  ├─ exp_seeding.yaml
│  └─ exp_assumption_violation.yaml
│
├─ notebooks/                        ← Jupyter notebooks for testing + plotting
│  ├─ 00_sanity_checks.ipynb
│  └─ 01_quick_plots.ipynb
│
├─ results/                          ← output from runs (NOT hand-edited)
│  └─ .gitkeep                       ← keeps folder in git even when empty
│
└─ scripts/                          ← easy command-line entry points
   ├─ run_experiment.py              ← runs a single config
   └─ run_sweep.py                   ← runs a sweep
```

# Pipeline

```
configs/*.yaml
     │
     ▼
scripts/run_experiment.py
     │
     ▼
experiments/runner.py
     │
     ├── networks/generators.py   (build graph G)
     ├── model/abm.py             (simulate steps)
     │      ├── model/agents.py
     │      ├── model/exchange.py
     │      ├── model/interventions.py
     │      └── model/shocks.py (optional)
     │
     ├── eval/measures.py         (compute metrics)
     └── utils/io.py              (save outputs)
     │
     ▼
results/<experiment_name>/...
```