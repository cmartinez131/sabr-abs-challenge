"""
Statistical Analysis for ABS Challenge Strategy

Generates ALL presentation charts from q_learning.py outputs:
- challenge_behavior_by_inning.png
- statistical_significance.png
- success_rate_comparison.png
- challenge_frequency.png
- projected_wins.png
- reward_comparison.png

Run this AFTER q_learning.py to generate charts from the saved CSV files.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
from collections import defaultdict
from q_learning import StateEncoder, GreedyPolicy, QLearningPolicy

OUTPUT_DIR = '../results/figures'

    
def load_model_and_data():
    """Load trained Q-table and pitch data."""
    model_paths = ['../models/q_table.pkl', 'models/q_table.pkl']
    data_paths = ['../data/processed/challengeable_pitches.csv', 
                  'data/processed/challengeable_pitches.csv']
    
    q_table = None
    inning_config = '4bucket_alt2'
    
    for path in model_paths:
        if os.path.exists(path):
            with open(path, 'rb') as f:
                model_data = pickle.load(f)
            q_table = model_data['q_table']
            inning_config = model_data.get('inning_config', '4bucket_alt2')
            print(f"Loaded Q-table: {len(q_table)} states")
            break
    
    if q_table is None:
        raise FileNotFoundError("Q-table not found. Run q_learning.py first.")
    
    df = None
    for path in data_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"Loaded {len(df):,} pitches")
            break
    
    if df is None:
        raise FileNotFoundError("Pitch data not found.")
    
    return q_table, df, inning_config


def load_results():
    """Load policy comparison results from CSV."""
    paths = ['../results/policy_comparison.csv', 'results/policy_comparison.csv']
    
    for path in paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            results = {}
            for _, row in df.iterrows():
                results[row['Policy']] = {
                    'mean': row['Avg_ER_Game'],
                    'std': row['Std_Dev'],
                    'success_rate': row['Success_Rate'],
                    'challenges': row['Challenges_Game'],
                    'projected_wins': row['Projected_Wins']
                }
            return results
    return None


def analyze_challenge_behavior(df, q_table, encoder):
    """Analyze challenge rate by inning for each policy."""
    print("\nAnalyzing challenge behavior by inning...")
    
    greedy = GreedyPolicy(lambda_penalty=0.02)
    qlearning = QLearningPolicy(q_table, encoder)
    
    results = {
        'Greedy': defaultdict(lambda: {'total': 0, 'challenges': 0}),
        'Q-Learning': defaultdict(lambda: {'total': 0, 'challenges': 0})
    }
    
    sample_size = min(50000, len(df))
    sample_df = df.sample(n=sample_size, random_state=42)
    
    for _, pitch in sample_df.iterrows():
        inning = min(int(pitch['inning']), 12)
        challenges_left = 2
        
        greedy_decision = greedy.decide(pitch, challenges_left)
        ql_decision = qlearning.decide(pitch, challenges_left)
        
        results['Greedy'][inning]['total'] += 1
        results['Greedy'][inning]['challenges'] += greedy_decision
        results['Q-Learning'][inning]['total'] += 1
        results['Q-Learning'][inning]['challenges'] += ql_decision
    
    innings = sorted([i for i in results['Greedy'].keys() if i <= 9])
    challenge_rates = {'Greedy': [], 'Q-Learning': [], 'innings': innings}
    
    print(f"\n{'Inning':<8} {'Greedy':<12} {'Q-Learning':<12} {'Diff':>10}")
    
    for inning in innings:
        greedy_rate = results['Greedy'][inning]['challenges'] / results['Greedy'][inning]['total'] * 100
        ql_rate = results['Q-Learning'][inning]['challenges'] / results['Q-Learning'][inning]['total'] * 100
        
        challenge_rates['Greedy'].append(greedy_rate)
        challenge_rates['Q-Learning'].append(ql_rate)
        
        print(f"{inning:<8} {greedy_rate:>6.1f}%      {ql_rate:>6.1f}%      {ql_rate - greedy_rate:>+6.1f}%")
    
    early_ql = np.mean([results['Q-Learning'][i]['challenges']/results['Q-Learning'][i]['total']*100 
                        for i in range(1,5)])
    late_ql = np.mean([results['Q-Learning'][i]['challenges']/results['Q-Learning'][i]['total']*100 
                       for i in range(7,10)])
    
    print(f"\nQ-Learning: Early={early_ql:.1f}%, Late={late_ql:.1f}%")
    
    return challenge_rates


def plot_challenge_behavior(challenge_rates):
    """Generate challenge behavior by inning charts (slides 15-16)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    innings = challenge_rates['innings']
    greedy_rates = challenge_rates['Greedy']
    ql_rates = challenge_rates['Q-Learning']
    
    # Line chart
    axes[0].plot(innings, greedy_rates, 'b-o', linewidth=2, markersize=8, label='Greedy')
    axes[0].plot(innings, ql_rates, 'r-s', linewidth=2, markersize=8, label='Q-Learning')
    axes[0].set_xlabel('Inning', fontsize=12)
    axes[0].set_ylabel('Challenge Rate (%)', fontsize=12)
    axes[0].set_title('Challenge Rate by Inning\n(Q-Learning is highly selective)', fontsize=14, fontweight='bold')
    axes[0].legend(loc='upper left')
    axes[0].set_xticks(innings)
    axes[0].grid(True, alpha=0.3)
    axes[0].axvspan(0.5, 4.5, alpha=0.1, color='blue')
    axes[0].axvspan(6.5, 9.5, alpha=0.1, color='red')
    
    # Difference chart
    diff = [q - g for q, g in zip(ql_rates, greedy_rates)]
    colors = ['#3498db' for _ in diff]
    
    bars = axes[1].bar(innings, diff, color=colors, edgecolor='black', alpha=0.8)
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axes[1].set_xlabel('Inning', fontsize=12)
    axes[1].set_ylabel('Q-Learning - Greedy (%)', fontsize=12)
    axes[1].set_title('Q-Learning Challenge Rate Difference\n(Negative = more selective)', fontsize=14, fontweight='bold')
    axes[1].set_xticks(innings)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, diff):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height - 2,
                    f'{val:.1f}%', ha='center', va='top', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/challenge_behavior_by_inning.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved challenge_behavior_by_inning.png")


def plot_policy_comparison(results):
    """Generate policy comparison with CIs (slide 19)."""
    if results is None:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    policies = [p for p in ['Terminal-Count', 'Greedy', 'Q-Learning'] if p in results]
    colors = ['#5DA5DA', '#60BD68', '#F17CB0'][:len(policies)]
    x = np.arange(len(policies))
    
    means = [results[p]['mean'] for p in policies]
    stds = [results[p]['std'] for p in policies]
    
    axes[0].bar(x, means, yerr=stds, capsize=5, color=colors, edgecolor='black', alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(policies)
    axes[0].set_ylabel('Expected Runs per Game')
    axes[0].set_title('Policy Comparison with 95% CIs\n(Non-overlapping = Significant)')
    
    for i, m in enumerate(means):
        axes[0].text(i, m + stds[i] + 0.005, f'{m:.4f}', ha='center', fontsize=10)
    
    wins = [results[p]['projected_wins'] for p in policies]
    axes[1].bar(x, wins, color=colors, edgecolor='black', alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(policies)
    axes[1].set_ylabel('Projected Wins per Season')
    axes[1].set_title('Real-World Impact:\nAdditional Wins from Challenge Strategy')
    
    for i, w in enumerate(wins):
        axes[1].text(i, w + 0.02, f'+{w:.2f}', ha='center', fontsize=10, fontweight='bold')
    
    if 'Q-Learning' in results and 'Greedy' in results:
        improvement = results['Q-Learning']['projected_wins'] - results['Greedy']['projected_wins']
        q_idx = policies.index('Q-Learning')
        axes[1].annotate(f'+{improvement:.2f} wins\nvs Greedy', 
                        xy=(q_idx, wins[q_idx]), xytext=(q_idx + 0.4, wins[q_idx] - 0.3),
                        fontsize=10, color='green', fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color='green'))
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/statistical_significance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved statistical_significance.png")


def plot_success_rates(results):
    """Generate success rate chart (slide 17)."""
    if results is None:
        return
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    policies = [p for p in ['Terminal-Count', 'Greedy', 'Q-Learning'] if p in results]
    colors = ['#5DA5DA', '#60BD68', '#F17CB0'][:len(policies)]
    rates = [results[p]['success_rate'] for p in policies]
    
    bars = ax.bar(policies, rates, color=colors, edgecolor='black', alpha=0.85)
    ax.set_ylabel('Challenge Success Rate (%)')
    ax.set_title('Challenge Success Rates')
    
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{rate:.1f}%', ha='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/success_rate_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved success_rate_comparison.png")


def plot_challenge_frequency(results):
    """Generate challenge frequency chart (slide 14)."""
    if results is None:
        return
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    policies = [p for p in ['Terminal-Count', 'Greedy', 'Q-Learning'] if p in results]
    colors = ['#5DA5DA', '#60BD68', '#F17CB0'][:len(policies)]
    freqs = [results[p]['challenges'] for p in policies]
    
    bars = ax.bar(policies, freqs, color=colors, edgecolor='black', alpha=0.85)
    ax.set_ylabel('Challenges per Game')
    ax.set_title('Challenge Frequency')
    ax.set_ylim(0, max(freqs) * 1.2)
    
    for bar, freq in zip(bars, freqs):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{freq:.2f}', ha='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/challenge_frequency.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved challenge_frequency.png")


def plot_projected_wins(results):
    """Generate projected wins chart (slide 18)."""
    if results is None:
        return
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    policies = [p for p in ['Terminal-Count', 'Greedy', 'Q-Learning'] if p in results]
    colors = ['#5DA5DA', '#60BD68', '#F17CB0'][:len(policies)]
    wins = [results[p]['projected_wins'] for p in policies]
    
    bars = ax.bar(policies, wins, color=colors, edgecolor='black', alpha=0.85)
    ax.set_ylabel('Projected Wins per Season')
    ax.set_title('Real-World Impact')
    ax.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, wins):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'+{val:.2f}', ha='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/projected_wins.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved projected_wins.png")


def plot_reward_comparison(results):
    """Generate reward comparison chart."""
    if results is None:
        return
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    policies = [p for p in ['Terminal-Count', 'Greedy', 'Q-Learning'] if p in results]
    colors = ['#5DA5DA', '#60BD68', '#F17CB0'][:len(policies)]
    rewards = [results[p]['mean'] for p in policies]
    
    bars = ax.bar(policies, rewards, color=colors, edgecolor='black', alpha=0.85)
    ax.set_ylabel('Average Expected Runs per Game')
    ax.set_title('Policy Performance Comparison')
    ax.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, rewards):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.002,
                f'{val:.4f}', ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/reward_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved reward_comparison.png")


def main():
    print("Statistical Analysis for ABS Challenge Strategy\n")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    try:
        q_table, df, inning_config = load_model_and_data()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    encoder = StateEncoder(inning_config=inning_config)
    print(f"State space: {encoder.num_states} states")
    
    # Challenge behavior analysis (requires Q-table and pitch data)
    challenge_rates = analyze_challenge_behavior(df, q_table, encoder)
    plot_challenge_behavior(challenge_rates)
    
    # Charts from policy_comparison.csv
    results = load_results()
    if results:
        print(f"\nPolicy results:")
        for p, r in results.items():
            print(f"  {p}: {r['mean']:.4f} ER/game, {r['success_rate']:.1f}% success, +{r['projected_wins']:.2f} wins")
        
        plot_policy_comparison(results)
        plot_success_rates(results)
        plot_challenge_frequency(results)
        plot_projected_wins(results)
        plot_reward_comparison(results)
    
    print(f"\nDone. All figures saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()