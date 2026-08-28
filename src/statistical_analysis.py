"""
Statistical Analysis for ABS Challenge Strategy

Generates ALL presentation charts from q_learning.py outputs:
- challenge_behavior_by_inning.png
- statistical_significance.png
- success_rate_comparison.png
- challenge_frequency.png
- projected_wins.png
- reward_comparison.png
- bootstrap_ci.png (NEW)

Also runs bootstrap significance testing to produce:
- p-value, Cohen's d, 95% CIs (cited in presentation)

Run this AFTER q_learning.py to generate charts from the saved CSV files.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
from collections import defaultdict
from q_learning import StateEncoder, GreedyPolicy, QLearningPolicy, SituationalThresholdPolicy, ConservativeCoachPolicy

# Auto-detect project root: works whether you run from src/ or project root
if os.path.exists('../results') and os.path.basename(os.getcwd()) == 'src':
    OUTPUT_DIR = '../results/figures'
    RESULTS_DIR = '../results'
else:
    OUTPUT_DIR = 'results/figures'
    RESULTS_DIR = 'results'

    
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
    """Analyze challenge rate by inning for ALL THREE policies."""
    print("\nAnalyzing challenge behavior by inning...")
    
    # Create all three policies
    situational = SituationalThresholdPolicy(threshold=0.08)
    greedy = GreedyPolicy(lambda_penalty=0.02)
    conservative = ConservativeCoachPolicy()
    qlearning = QLearningPolicy(q_table, encoder)
    
    results = {
        'Situational-Threshold': defaultdict(lambda: {'total': 0, 'challenges': 0}),
        'Greedy': defaultdict(lambda: {'total': 0, 'challenges': 0}),
        'Conservative-Coach': defaultdict(lambda: {'total': 0, 'challenges': 0}),
        'Q-Learning': defaultdict(lambda: {'total': 0, 'challenges': 0})
    }
    
    sample_size = min(50000, len(df))
    sample_df = df.sample(n=sample_size, random_state=42)

    # Group by game so challenges deplete properly across innings
    if 'game_pk' not in sample_df.columns:
        print("WARNING: game_pk not found -- challenges_left not tracked per game.")
        for _, pitch in sample_df.iterrows():
            inning = min(int(pitch['inning']), 12)
            challenges_left = 2
            sit_decision = situational.decide(pitch, challenges_left)
            greedy_decision = greedy.decide(pitch, challenges_left)
            cons_decision = conservative.decide(pitch, challenges_left)
            ql_decision = qlearning.decide(pitch, challenges_left)
            results['Situational-Threshold'][inning]['total'] += 1
            results['Situational-Threshold'][inning]['challenges'] += sit_decision
            results['Greedy'][inning]['total'] += 1
            results['Greedy'][inning]['challenges'] += greedy_decision
            results['Conservative-Coach'][inning]['total'] += 1
            results['Conservative-Coach'][inning]['challenges'] += cons_decision
            results['Q-Learning'][inning]['total'] += 1
            results['Q-Learning'][inning]['challenges'] += ql_decision
    else:
        for game_pk, game_df in sample_df.groupby('game_pk'):
            game_df = game_df.sort_values('inning')
            # Each policy gets its own independent challenge budget per game
            sit_challenges = 2
            greedy_challenges = 2
            cons_challenges = 2
            ql_challenges = 2

            for _, pitch in game_df.iterrows():
                inning = min(int(pitch['inning']), 12)

                sit_decision = situational.decide(pitch, sit_challenges)
                greedy_decision = greedy.decide(pitch, greedy_challenges)
                cons_decision = conservative.decide(pitch, cons_challenges)
                ql_decision = qlearning.decide(pitch, ql_challenges)

                # Only lose challenge on failure (keep it if call is overturned)
                is_incorrect = bool(pitch.get('call_incorrect', False))
                if not is_incorrect:
                    if sit_decision == 1:
                        sit_challenges = max(0, sit_challenges - 1)
                    if greedy_decision == 1:
                        greedy_challenges = max(0, greedy_challenges - 1)
                    if cons_decision == 1:
                        cons_challenges = max(0, cons_challenges - 1)
                    if ql_decision == 1:
                        ql_challenges = max(0, ql_challenges - 1)

                results['Situational-Threshold'][inning]['total'] += 1
                results['Situational-Threshold'][inning]['challenges'] += sit_decision
                results['Greedy'][inning]['total'] += 1
                results['Greedy'][inning]['challenges'] += greedy_decision
                results['Conservative-Coach'][inning]['total'] += 1
                results['Conservative-Coach'][inning]['challenges'] += cons_decision
                results['Q-Learning'][inning]['total'] += 1
                results['Q-Learning'][inning]['challenges'] += ql_decision
    
    innings = sorted([i for i in results['Greedy'].keys() if i <= 9])
    challenge_rates = {
        'Situational-Threshold': [],
        'Greedy': [],
        'Conservative-Coach': [],
        'Q-Learning': [],
        'innings': innings
    }
    
    print(f"\n{'Inning':<8} {'Sit-Thresh':<14} {'Greedy':<12} {'Conservative':<15} {'Q-Learning':<12}")
    
    for inning in innings:
        sit_rate = results['Situational-Threshold'][inning]['challenges'] / results['Situational-Threshold'][inning]['total'] * 100
        greedy_rate = results['Greedy'][inning]['challenges'] / results['Greedy'][inning]['total'] * 100
        ql_rate = results['Q-Learning'][inning]['challenges'] / results['Q-Learning'][inning]['total'] * 100
        
        challenge_rates['Situational-Threshold'].append(sit_rate)
        challenge_rates['Greedy'].append(greedy_rate)
        # Conservative-Coach appended in print block above
        challenge_rates['Q-Learning'].append(ql_rate)
        
        cons_rate = results['Conservative-Coach'][inning]['challenges'] / results['Conservative-Coach'][inning]['total'] * 100
        challenge_rates['Conservative-Coach'].append(cons_rate)
        print(f"{inning:<8} {sit_rate:>6.1f}%       {greedy_rate:>6.1f}%      {cons_rate:>6.1f}%         {ql_rate:>6.1f}%")
    
    # Summary stats
    for name in ['Situational-Threshold', 'Greedy', 'Conservative-Coach', 'Q-Learning']:
        early = np.mean([results[name][i]['challenges']/results[name][i]['total']*100 
                         for i in range(1,5)])
        late = np.mean([results[name][i]['challenges']/results[name][i]['total']*100 
                        for i in range(7,10)])
        print(f"{name}: Early(1-4)={early:.1f}%, Late(7-9)={late:.1f}%")
    
    return challenge_rates


def plot_challenge_behavior(challenge_rates):
    """Generate challenge behavior by inning charts with ALL THREE policies."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    innings = challenge_rates['innings']
    sit_rates = challenge_rates['Situational-Threshold']
    greedy_rates = challenge_rates['Greedy']
    cons_rates = challenge_rates.get('Conservative-Coach', [0]*len(innings))
    ql_rates = challenge_rates['Q-Learning']
    
    # Line chart - all three policies
    axes[0].plot(innings, sit_rates, 'g-^', linewidth=2, markersize=8, label='Situational Threshold')
    axes[0].plot(innings, greedy_rates, 'b-o', linewidth=2, markersize=8, label='Greedy')
    axes[0].plot(innings, cons_rates, 'orange', marker='D', linewidth=2, markersize=8, label='Conservative-Coach')
    axes[0].plot(innings, ql_rates, 'r-s', linewidth=2, markersize=8, label='Q-Learning')
    axes[0].set_xlabel('Inning', fontsize=12)
    axes[0].set_ylabel('Challenge Rate (%)', fontsize=12)
    axes[0].set_title('Challenge Rate by Inning\n(All Three Policies)', fontsize=14, fontweight='bold')
    axes[0].legend(loc='best')
    axes[0].set_xticks(innings)
    axes[0].grid(True, alpha=0.3)
    axes[0].axvspan(0.5, 4.5, alpha=0.05, color='blue', label='_nolegend_')
    axes[0].axvspan(6.5, 9.5, alpha=0.05, color='red', label='_nolegend_')
    
    # Difference chart - Q-Learning vs each baseline
    x = np.arange(len(innings))
    width = 0.35
    diff_sit = [q - s for q, s in zip(ql_rates, sit_rates)]
    diff_greedy = [q - g for q, g in zip(ql_rates, greedy_rates)]
    diff_cons = [q - c for q, c in zip(ql_rates, cons_rates)]
    
    width = 0.25
    bars1 = axes[1].bar(x - width, diff_sit, width, color='#2ecc71', edgecolor='black',
                         alpha=0.8, label='vs Sit. Threshold')
    bars2 = axes[1].bar(x, diff_greedy, width, color='#3498db', edgecolor='black',
                         alpha=0.8, label='vs Greedy')
    bars3 = axes[1].bar(x + width, diff_cons, width, color='orange', edgecolor='black',
                         alpha=0.8, label='vs Conservative-Coach')
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axes[1].set_xlabel('Inning', fontsize=12)
    axes[1].set_ylabel('Q-Learning - Baseline (%)', fontsize=12)
    axes[1].set_title('Q-Learning Challenge Rate Difference\n(Negative = more selective)', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(innings)
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/challenge_behavior_by_inning.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved challenge_behavior_by_inning.png")


def run_bootstrap_analysis(n_iterations=10000):
    """
    Bootstrap significance testing.
    
    Produces the p-value, Cohen's d, and 95% CIs cited in the presentation.
    Compares Q-Learning vs Greedy and Q-Learning vs Situational-Threshold.
    
    Requires per_game_results.csv from q_learning.py.
    """
    paths = ['../results/per_game_results.csv', 'results/per_game_results.csv']
    
    df = None
    for path in paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            break
    
    if df is None:
        print("WARNING: per_game_results.csv not found. Skipping bootstrap.")
        print("  (Run q_learning.py first to generate this file)")
        return None
    
    print(f"\n{'='*60}")
    print(f"BOOTSTRAP SIGNIFICANCE TESTING ({n_iterations:,} iterations)")
    print(f"{'='*60}")
    
    # Get per-game rewards for each policy
    ql_rewards = df[df['policy'] == 'Q-Learning']['reward'].values
    greedy_rewards = df[df['policy'] == 'Greedy']['reward'].values
    sit_rewards = df[df['policy'] == 'Situational-Threshold']['reward'].values
    cons_rewards = df[df['policy'] == 'Conservative-Coach']['reward'].values
    
    print(f"\nGames per policy:")
    print(f"  Q-Learning:             {len(ql_rewards)}")
    print(f"  Greedy:                 {len(greedy_rewards)}")
    print(f"  Situational-Threshold:  {len(sit_rewards)}")
    print(f"  Conservative-Coach:     {len(cons_rewards)}")
    
    all_results = {}
    
    # Compare Q-Learning vs each baseline
    comparisons = [
        ('Q-Learning', 'Greedy', ql_rewards, greedy_rewards),
        ('Q-Learning', 'Situational-Threshold', ql_rewards, sit_rewards),
        ('Q-Learning', 'Conservative-Coach', ql_rewards, cons_rewards),
    ]
    
    np.random.seed(42)
    
    for name_a, name_b, rewards_a, rewards_b in comparisons:
        if len(rewards_a) == 0 or len(rewards_b) == 0:
            print(f"\nWARNING: Missing data for {name_a} vs {name_b}")
            continue
        
        print(f"\n--- {name_a} vs {name_b} ---")
        
        # Observed stats
        observed_diff = np.mean(rewards_a) - np.mean(rewards_b)
        print(f"  {name_a} mean:  {np.mean(rewards_a):.4f} ER/game")
        print(f"  {name_b} mean:  {np.mean(rewards_b):.4f} ER/game")
        print(f"  Observed diff:  {observed_diff:.4f} ER/game")
        
        # Bootstrap resampling
        boot_diffs = []
        for _ in range(n_iterations):
            sample_a = np.random.choice(rewards_a, size=len(rewards_a), replace=True)
            sample_b = np.random.choice(rewards_b, size=len(rewards_b), replace=True)
            boot_diffs.append(np.mean(sample_a) - np.mean(sample_b))
        
        boot_diffs = np.array(boot_diffs)
        
        # p-value: proportion of bootstrap samples where diff <= 0
        p_value = np.mean(boot_diffs <= 0)
        
        # 95% confidence interval on the difference
        ci_lower = np.percentile(boot_diffs, 2.5)
        ci_upper = np.percentile(boot_diffs, 97.5)
        
        # Cohen's d (effect size)
        pooled_std = np.sqrt((np.std(rewards_a)**2 + np.std(rewards_b)**2) / 2)
        cohens_d = observed_diff / pooled_std if pooled_std > 0 else 0
        
        # Convert to wins
        wins_diff = observed_diff * 162 / 10
        wins_ci_lower = ci_lower * 162 / 10
        wins_ci_upper = ci_upper * 162 / 10
        
        p_str = f"{p_value:.4f}" if p_value >= 0.001 else "< 0.001"
        
        print(f"\n  Results:")
        print(f"    p-value:         {p_str}")
        print(f"    95% CI (ER/g):   [{ci_lower:.4f}, {ci_upper:.4f}]")
        print(f"    95% CI (wins):   [{wins_ci_lower:.1f}, {wins_ci_upper:.1f}]")
        
        all_results[f'{name_a}_vs_{name_b}'] = {
            'observed_diff': observed_diff,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'wins_diff': wins_diff,
            'wins_ci_lower': wins_ci_lower,
            'wins_ci_upper': wins_ci_upper,
            'boot_diffs': boot_diffs
        }
    
    # Also compute absolute CI for Q-Learning ER/game
    boot_ql_means = []
    for _ in range(n_iterations):
        sample = np.random.choice(ql_rewards, size=len(ql_rewards), replace=True)
        boot_ql_means.append(np.mean(sample))
    
    ql_ci_lower = np.percentile(boot_ql_means, 2.5)
    ql_ci_upper = np.percentile(boot_ql_means, 97.5)
    print(f"\nQ-Learning absolute 95% CI: [{ql_ci_lower:.4f}, {ql_ci_upper:.4f}] ER/game")
    all_results['ql_absolute_ci'] = (ql_ci_lower, ql_ci_upper)
    
    # Save results to file
    try:
        results_dir = RESULTS_DIR
        os.makedirs(results_dir, exist_ok=True)
        with open(f'{results_dir}/bootstrap_results.txt', 'w') as f:
            f.write(f"BOOTSTRAP SIGNIFICANCE TESTING\n")
            f.write(f"Iterations: {n_iterations:,}\n")
            f.write(f"Seed: 42\n\n")
            
            for key, res in all_results.items():
                if key == 'ql_absolute_ci':
                    f.write(f"\nQ-Learning absolute 95% CI: [{res[0]:.4f}, {res[1]:.4f}]\n")
                    continue
                f.write(f"{key}:\n")
                f.write(f"  Observed diff: {res['observed_diff']:.4f} ER/game\n")
                p_str = f"{res['p_value']:.4f}" if res['p_value'] >= 0.001 else "< 0.001"
                f.write(f"  p-value: {p_str}\n")
                f.write(f"  Cohen's d: {res['cohens_d']:.2f}\n")
                f.write(f"  95% CI (ER/game): [{res['ci_lower']:.4f}, {res['ci_upper']:.4f}]\n")
                f.write(f"  95% CI (wins/szn): [{res['wins_ci_lower']:.1f}, {res['wins_ci_upper']:.1f}]\n\n")
        
        print(f"\nSaved bootstrap_results.txt")
    except Exception as e:
        print(f"Could not save bootstrap results: {e}")
    
    # Plot bootstrap distributions
    plot_bootstrap(all_results)
    
    return all_results


def plot_bootstrap(bootstrap_results):
    """Plot bootstrap distributions for presentation."""
    comparisons = [k for k in bootstrap_results.keys() if k != 'ql_absolute_ci']
    
    if not comparisons:
        return
    
    fig, axes = plt.subplots(1, len(comparisons), figsize=(6 * len(comparisons), 5))
    if len(comparisons) == 1:
        axes = [axes]
    
    for ax, key in zip(axes, comparisons):
        res = bootstrap_results[key]
        diffs = res['boot_diffs']
        
        ax.hist(diffs, bins=80, color='#3498db', edgecolor='black', alpha=0.7, density=True)
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No difference')
        ax.axvline(x=res['observed_diff'], color='green', linestyle='-', linewidth=2, 
                   label=f'Observed ({res["observed_diff"]:.4f})')
        ax.axvline(x=res['ci_lower'], color='orange', linestyle=':', linewidth=1.5)
        ax.axvline(x=res['ci_upper'], color='orange', linestyle=':', linewidth=1.5, 
                   label=f'95% CI [{res["ci_lower"]:.4f}, {res["ci_upper"]:.4f}]')
        
        label = key.replace('_vs_', ' vs ')
        p_str = f"p{res['p_value']:.4f}" if res['p_value'] >= 0.001 else "p < 0.001"
        ax.set_title(f'{label}\n{p_str}', 
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('Difference in ER/Game', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/bootstrap_ci.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved bootstrap_ci.png")


def plot_policy_comparison(results):
    """Generate policy comparison with CIs (slide 19)."""
    if results is None:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    policies = [p for p in ['Situational-Threshold', 'Greedy', 'Conservative-Coach', 'Q-Learning'] if p in results]
    colors = ['#5DA5DA', '#60BD68', '#F5A623', '#F17CB0'][:len(policies)]
    x = np.arange(len(policies))
    
    means = [results[p]['mean'] for p in policies]
    stds = [results[p]['std'] for p in policies]
    
    axes[0].bar(x, means, yerr=stds, capsize=5, color=colors, edgecolor='black', alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(policies, fontsize=8)
    axes[0].set_ylabel('Expected Runs per Game')
    axes[0].set_title('Policy Comparison with 95% CIs\n(Non-overlapping = Significant)')
    
    for i, m in enumerate(means):
        axes[0].text(i, m + stds[i] + 0.005, f'{m:.4f}', ha='center', fontsize=10)
    
    wins = [results[p]['projected_wins'] for p in policies]
    axes[1].bar(x, wins, color=colors, edgecolor='black', alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(policies, fontsize=8)
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
    
    policies = [p for p in ['Situational-Threshold', 'Greedy', 'Conservative-Coach', 'Q-Learning'] if p in results]
    colors = ['#5DA5DA', '#60BD68', '#F5A623', '#F17CB0'][:len(policies)]
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
    
    policies = [p for p in ['Situational-Threshold', 'Greedy', 'Conservative-Coach', 'Q-Learning'] if p in results]
    colors = ['#5DA5DA', '#60BD68', '#F5A623', '#F17CB0'][:len(policies)]
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
    
    policies = [p for p in ['Situational-Threshold', 'Greedy', 'Conservative-Coach', 'Q-Learning'] if p in results]
    colors = ['#5DA5DA', '#60BD68', '#F5A623', '#F17CB0'][:len(policies)]
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
    
    policies = [p for p in ['Situational-Threshold', 'Greedy', 'Conservative-Coach', 'Q-Learning'] if p in results]
    colors = ['#5DA5DA', '#60BD68', '#F5A623', '#F17CB0'][:len(policies)]
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
    print("Statistical Analysis for ABS Challenge Strategy")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    try:
        q_table, df, inning_config = load_model_and_data()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    encoder = StateEncoder(inning_config=inning_config)
    print(f"State space: {encoder.num_states} states")
    
    # 1. Challenge behavior analysis (all 3 policies, by inning)
    print("\n[1/3] Challenge behavior by inning...")
    challenge_rates = analyze_challenge_behavior(df, q_table, encoder)
    plot_challenge_behavior(challenge_rates)
    
    # 2. Charts from policy_comparison.csv (all 3 policies)
    print("\n[2/3] Policy comparison charts...")
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
    else:
        print("WARNING: policy_comparison.csv not found. Skipping comparison charts.")
    
    # 3. Bootstrap significance testing
    print("\n[3/3] Bootstrap significance testing...")
    bootstrap = run_bootstrap_analysis(n_iterations=10000)
    
    print(f"\n{'='*60}")
    print(f"COMPLETE. All figures saved to {OUTPUT_DIR}/")
    print(f"{'='*60}")
    
    # Print summary for easy copy-paste into presentation
    if bootstrap and results:
        print("\n NUMBERS FOR PRESENTATION ")
        if 'Q-Learning' in results:
            print(f"Q-Learning: {results['Q-Learning']['mean']:.4f} ER/game, "
                  f"+{results['Q-Learning']['projected_wins']:.2f} wins/season")
        if 'Greedy' in results:
            print(f"Greedy:     {results['Greedy']['mean']:.4f} ER/game, "
                  f"+{results['Greedy']['projected_wins']:.2f} wins/season")
        if 'Situational-Threshold' in results:
            print(f"Sit-Thresh: {results['Situational-Threshold']['mean']:.4f} ER/game, "
                  f"+{results['Situational-Threshold']['projected_wins']:.2f} wins/season")
        if 'Conservative-Coach' in results:
            print(f"Cons-Coach: {results['Conservative-Coach']['mean']:.4f} ER/game, "
                  f"+{results['Conservative-Coach']['projected_wins']:.2f} wins/season")
        
        for key_name, label in [
            ('Q-Learning_vs_Greedy', 'Q-L vs Greedy'),
            ('Q-Learning_vs_Situational-Threshold', 'Q-L vs Sit-Thresh'),
            ('Q-Learning_vs_Conservative-Coach', 'Q-L vs Cons-Coach'),
        ]:
            comp = bootstrap.get(key_name)
            if comp:
                p_str = f"{comp['p_value']:.4f}" if comp['p_value'] >= 0.001 else "< 0.001"
                print(f"\n{label}: p = {p_str}, 95% CI [{comp['ci_lower']:.4f}, {comp['ci_upper']:.4f}]")



if __name__ == "__main__":
    main()