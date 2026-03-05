"""Quick analysis of experiment results."""
import pandas as pd
from scipy import stats
import sys

def analyze_topology():
    df = pd.read_csv('results/topology_sweep/runs_summary.csv')
    
    print("=" * 60)
    print("TOPOLOGY SWEEP RESULTS")
    print("=" * 60)
    
    # Per-topology stats
    for gt in ['er', 'ws', 'ba', 'sbm']:
        sub = df[df['graph_type'] == gt]
        print(f"\n{gt.upper()} (n={len(sub)} runs):")
        for col in ['gini_final', 'poverty_final', 'escape_fraction', 'escape_time_mean', 'spearman_rank_corr']:
            m, s = sub[col].mean(), sub[col].std()
            print(f"  {col:25s}: {m:.4f} ± {s:.4f}")
    
    # Kruskal-Wallis across all 4
    print("\n--- Kruskal-Wallis (all 4 topologies) ---")
    for col in ['gini_final', 'poverty_final', 'escape_fraction', 'spearman_rank_corr']:
        groups = [df[df['graph_type'] == g][col] for g in ['er', 'ws', 'ba', 'sbm']]
        h, p = stats.kruskal(*groups)
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        print(f"  {col:25s}: H={h:7.3f}, p={p:.6f} {sig}")
    
    # Pairwise BA vs WS
    ba = df[df['graph_type'] == 'ba']
    ws = df[df['graph_type'] == 'ws']
    print("\n--- BA vs WS (Mann-Whitney U) ---")
    for col in ['gini_final', 'poverty_final', 'escape_fraction', 'spearman_rank_corr']:
        u, p = stats.mannwhitneyu(ba[col], ws[col], alternative='two-sided')
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        print(f"  {col:25s}: BA={ba[col].mean():.4f}, WS={ws[col].mean():.4f}, p={p:.6f} {sig}")


def analyze_seeding():
    try:
        df = pd.read_csv('results/seeding_aid/runs_summary.csv')
    except Exception:
        print("\nSeeding results not yet available.")
        return
    
    print("\n" + "=" * 60)
    print("SEEDING EXPERIMENT RESULTS")
    print("=" * 60)
    
    strategies = df['strategy'].unique()
    print(f"Strategies: {list(strategies)}")
    print(f"Runs per strategy: {df.groupby('strategy').size().iloc[0]}")
    
    # Per-strategy stats
    cols = ['escape_fraction', 'gini_final', 'poverty_final', 'spearman_rank_corr',
            'treated_mean_dW', 'neighbor_mean_dW', 'non_treated_mean_dW']
    
    for strat in sorted(strategies):
        sub = df[df['strategy'] == strat]
        print(f"\n{strat}:")
        for col in cols:
            if col in sub.columns:
                m, s = sub[col].mean(), sub[col].std()
                print(f"  {col:30s}: {m:+.4f} ± {s:.4f}" if 'dW' in col else f"  {col:30s}: {m:.4f} ± {s:.4f}")
    
    # Compare each strategy vs "none" baseline
    if 'none' in strategies:
        print("\n--- Each strategy vs. NONE baseline (Mann-Whitney U) ---")
        none_df = df[df['strategy'] == 'none']
        for strat in sorted(strategies):
            if strat == 'none':
                continue
            strat_df = df[df['strategy'] == strat]
            for col in ['escape_fraction', 'poverty_final']:
                alt = 'greater' if col == 'escape_fraction' else 'less'
                u, p = stats.mannwhitneyu(strat_df[col], none_df[col], alternative=alt)
                sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
                diff = strat_df[col].mean() - none_df[col].mean()
                print(f"  {strat:25s} {col:20s}: diff={diff:+.4f}, p={p:.6f} {sig}")
    
    # Kruskal-Wallis across all strategies
    print("\n--- Kruskal-Wallis (all strategies) ---")
    for col in ['escape_fraction', 'poverty_final', 'gini_final']:
        groups = [df[df['strategy'] == s][col] for s in sorted(strategies)]
        h, p = stats.kruskal(*groups)
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        print(f"  {col:25s}: H={h:7.3f}, p={p:.6f} {sig}")


if __name__ == "__main__":
    analyze_topology()
    analyze_seeding()
