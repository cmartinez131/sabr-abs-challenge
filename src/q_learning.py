"""
Q-Learning for ABS Challenge Strategy

Optimizing when to challenge ball/strike calls using reinforcement learning.

DATA SOURCE: MLB Statcast pitch tracking (June-September 2024)
METHODOLOGY: We SIMULATE ABS challenge outcomes using pitch location as ground truth.
             This is NOT real Triple-A challenge data - we model P(overturn) from
             pitch location relative to the strike zone.

BEST CONFIGURATION (from hyperparameter experiment):
- 4bucket_alt2: Innings 1-4, 5-7, 8-9, 10+ 
- Typical: ~0.10 ER/game, ~+1.6-1.8 wins/season

OUTPUTS:
- models/q_table.pkl (trained Q-table)
- results/policy_comparison.csv (policy results)
- results/training_history.csv (training progress)
- results/per_game_results.csv (per-game data for statistical analysis)
- results/figures/training_progress.png

Run statistical_analysis.py after this to generate all presentation charts.
"""

import random
import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
from collections import defaultdict

# Reproducibility
SEED = 21
random.seed(SEED)
np.random.seed(SEED)


# Q-LEARNER CLASS


class QLearner:
    """Q-Learning implementation adapted from Georgia Tech ML4T course."""
    
    def __init__(
        self,
        num_states=100,
        num_actions=2,
        alpha=0.2,
        gamma=0.9,
        rar=0.5,
        radr=0.99,
        dyna=0,
        verbose=False,
    ):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.verbose = verbose
        
        self.s = 0
        self.a = 0
        
        self.Q = np.zeros((num_states, num_actions))
        self.experience = {}
    
    def querysetstate(self, s):
        """Set state and return action (for first step)."""
        self.s = s
        
        if random.random() < self.rar:
            action = random.randint(0, self.num_actions - 1)
        else:
            action = np.argmax(self.Q[s])
        
        self.a = action
        return action
    
    def query(self, s_prime, r):
        """Update Q-table and return next action."""
        prev_q = self.Q[self.s, self.a]
        max_future_q = np.max(self.Q[s_prime])
        
        new_q = (1 - self.alpha) * prev_q + self.alpha * (r + self.gamma * max_future_q)
        self.Q[self.s, self.a] = new_q
        
        key = (self.s, self.a)
        if key not in self.experience:
            self.experience[key] = []
        self.experience[key].append((s_prime, r))
        
        if self.dyna > 0 and len(self.experience) > 0:
            for _ in range(self.dyna):
                rand_key = random.choice(list(self.experience.keys()))
                rand_s, rand_a = rand_key
                rand_s_prime, rand_r = random.choice(self.experience[rand_key])
                
                prev_q = self.Q[rand_s, rand_a]
                max_future = np.max(self.Q[rand_s_prime])
                self.Q[rand_s, rand_a] = (
                    (1 - self.alpha) * prev_q + 
                    self.alpha * (rand_r + self.gamma * max_future)
                )
        
        self.s = s_prime
        
        if random.random() < self.rar:
            action = random.randint(0, self.num_actions - 1)
        else:
            action = np.argmax(self.Q[s_prime])
        
        self.a = action
        return action
    
    def decay_exploration(self):
        self.rar *= self.radr
    
    def get_action(self, s):
        return np.argmax(self.Q[s])


# STATE ENCODER


class StateEncoder:
    """Encode game context as discrete state."""
    
    INNING_CONFIGS = {
        '3bucket': {
            'name': '3-bucket (1-3, 4-6, 7+)',
            'n_buckets': 3,
            'buckets': lambda inn: 0 if inn <= 3 else (1 if inn <= 6 else 2),
            'labels': ['Early (1-3)', 'Mid (4-6)', 'Late (7+)']
        },
        '3bucket_alt1': {
            'name': '3-bucket Alt1 (1-6, 7-8, 9+)',
            'n_buckets': 3,
            'buckets': lambda inn: 0 if inn <= 6 else (1 if inn <= 8 else 2),
            'labels': ['Early (1-6)', 'Setup (7-8)', 'Late (9+)']
        },
        '4bucket': {
            'name': '4-bucket (1-6, 7-8, 9, 10+)',
            'n_buckets': 4,
            'buckets': lambda inn: 0 if inn <= 6 else (1 if inn <= 8 else (2 if inn == 9 else 3)),
            'labels': ['Marathon (1-6)', 'Setup (7-8)', 'Ninth (9)', 'Extras (10+)']
        },
        '4bucket_alt1': {
            'name': '4-bucket Alt1 (1-5, 6-7, 8-9, 10+)',
            'n_buckets': 4,
            'buckets': lambda inn: 0 if inn <= 5 else (1 if inn <= 7 else (2 if inn <= 9 else 3)),
            'labels': ['Early (1-5)', 'Mid (6-7)', 'Late (8-9)', 'Extras (10+)']
        },
        '4bucket_alt2': {
            'name': '4-bucket Alt2 (1-4, 5-7, 8-9, 10+)',
            'n_buckets': 4,
            'buckets': lambda inn: 0 if inn <= 4 else (1 if inn <= 7 else (2 if inn <= 9 else 3)),
            'labels': ['Early (1-4)', 'Mid (5-7)', 'Late (8-9)', 'Extras (10+)']
        },
        '5bucket': {
            'name': '5-bucket (1-5, 6, 7-8, 9, 10+)',
            'n_buckets': 5,
            'buckets': lambda inn: 0 if inn <= 5 else (1 if inn == 6 else (2 if inn <= 8 else (3 if inn == 9 else 4))),
            'labels': ['Early (1-5)', 'Sixth', 'Setup (7-8)', 'Ninth', 'Extras (10+)']
        }
    }
    
    def __init__(self, use_full_state=False, inning_config='4bucket_alt2'):
        self.use_full_state = use_full_state
        self.inning_config = inning_config
        
        if inning_config not in self.INNING_CONFIGS:
            inning_config = '4bucket_alt2'
        
        self.inning_setup = self.INNING_CONFIGS[inning_config]
        self.n_inning_buckets = self.inning_setup['n_buckets']
        self.inning_encoder = self.inning_setup['buckets']
        
        print(f"Inning configuration: {self.inning_setup['name']}")
        print(f"  Buckets: {self.inning_setup['labels']}")
        
        if use_full_state:
            self.dims = {
                'inning': self.n_inning_buckets, 
                'outs': 3, 'balls': 4, 'strikes': 3,
                'runners': 8, 'score_diff': 7, 'challenges': 3, 'distance': 3
            }
        else:
            self.dims = {
                'inning': self.n_inning_buckets,
                'outs': 3,
                'count': 4,
                'runners': 3,
                'score': 3,
                'challenges': 3,
                'distance': 2
            }
        
        self.num_states = 1
        for dim in self.dims.values():
            self.num_states *= dim
        
        print(f"State space size: {self.num_states:,} discrete states")
        if not use_full_state:
            print(f"  (Using simplified state space for faster convergence)")
    
    def encode(self, pitch_data, challenges_left):
        def get_val(key, default=0):
            val = pitch_data.get(key, default) if isinstance(pitch_data, dict) else pitch_data.get(key, default)
            return default if pd.isna(val) else val
        
        inning = int(get_val('inning', 1))
        inning_idx = self.inning_encoder(inning)
        outs_idx = min(int(get_val('outs_when_up', 0)), 2)
        
        if self.use_full_state:
            balls_idx = min(int(get_val('balls', 0)), 3)
            strikes_idx = min(int(get_val('strikes', 0)), 2)
            
            on_1b = 1 if pd.notna(pitch_data.get('on_1b')) and pitch_data.get('on_1b') != 0 else 0
            on_2b = 1 if pd.notna(pitch_data.get('on_2b')) and pitch_data.get('on_2b') != 0 else 0
            on_3b = 1 if pd.notna(pitch_data.get('on_3b')) and pitch_data.get('on_3b') != 0 else 0
            runners_idx = on_1b + 2 * on_2b + 4 * on_3b
            
            score_diff = get_val('score_diff', 0)
            if score_diff <= -3:
                score_idx = 0
            elif score_diff == -2:
                score_idx = 1
            elif score_diff == -1:
                score_idx = 2
            elif score_diff == 0:
                score_idx = 3
            elif score_diff == 1:
                score_idx = 4
            elif score_diff == 2:
                score_idx = 5
            else:
                score_idx = 6
            
            chal_idx = min(int(challenges_left), 2)
            
            distance = get_val('dist_from_zone', 0.15)
            if distance < 0.083:
                dist_idx = 0
            elif distance < 0.167:
                dist_idx = 1
            else:
                dist_idx = 2
            
            state = (
                inning_idx * (3 * 4 * 3 * 8 * 7 * 3 * 3) +
                outs_idx * (4 * 3 * 8 * 7 * 3 * 3) +
                balls_idx * (3 * 8 * 7 * 3 * 3) +
                strikes_idx * (8 * 7 * 3 * 3) +
                runners_idx * (7 * 3 * 3) +
                score_idx * (3 * 3) +
                chal_idx * 3 +
                dist_idx
            )
        else:
            balls = int(get_val('balls', 0))
            strikes = int(get_val('strikes', 0))
            if strikes == 2:
                count_idx = 3
            elif balls > strikes:
                count_idx = 0
            elif strikes > balls:
                count_idx = 2
            else:
                count_idx = 1
            
            on_1b = 1 if pd.notna(pitch_data.get('on_1b')) and pitch_data.get('on_1b') != 0 else 0
            on_2b = 1 if pd.notna(pitch_data.get('on_2b')) and pitch_data.get('on_2b') != 0 else 0
            on_3b = 1 if pd.notna(pitch_data.get('on_3b')) and pitch_data.get('on_3b') != 0 else 0
            
            if on_2b or on_3b:
                runners_idx = 1
            elif on_1b:
                runners_idx = 2
            else:
                runners_idx = 0
            
            score_diff = get_val('score_diff', 0)
            if score_diff < 0:
                score_idx = 0
            elif score_diff == 0:
                score_idx = 1
            else:
                score_idx = 2
            
            chal_idx = min(int(challenges_left), 2)
            
            distance = get_val('dist_from_zone', 0.15)
            dist_idx = 0 if distance < 0.167 else 1
            
            state = (
                inning_idx * (3 * 4 * 3 * 3 * 3 * 2) +
                outs_idx * (4 * 3 * 3 * 3 * 2) +
                count_idx * (3 * 3 * 3 * 2) +
                runners_idx * (3 * 3 * 2) +
                score_idx * (3 * 2) +
                chal_idx * 2 +
                dist_idx
            )
        
        return min(state, self.num_states - 1)


# BASELINE POLICIES


class TerminalCountPolicy:
    """Baseline A: Challenge on terminal counts near zone edge."""
    
    def __init__(self, threshold=0.08):
        self.name = "Terminal-Count"
        self.threshold = threshold
    
    def decide(self, pitch_data, challenges_left):
        if challenges_left <= 0:
            return 0
        
        score = 0.0
        
        distance = pitch_data.get('dist_from_zone', 0.25)
        if pd.isna(distance):
            distance = 0.25
        
        dist_inches = distance * 12
        if dist_inches < 1.0:
            score += 0.20
        elif dist_inches < 2.0:
            score += 0.10
        elif dist_inches < 3.0:
            score += 0.02
        
        inning = pitch_data.get('inning', 1)
        if inning >= 9:
            score += 0.08
        elif inning >= 7:
            score += 0.04
        elif inning >= 4:
            score += 0.01
        
        outs = pitch_data.get('outs_when_up', 0)
        if outs == 2:
            score += 0.05
        elif outs == 1:
            score += 0.01
        
        on_2b = pd.notna(pitch_data.get('on_2b')) and pitch_data.get('on_2b') != 0
        on_3b = pd.notna(pitch_data.get('on_3b')) and pitch_data.get('on_3b') != 0
        on_1b = pd.notna(pitch_data.get('on_1b')) and pitch_data.get('on_1b') != 0
        
        if on_3b:
            score += 0.06
        if on_2b:
            score += 0.04
        if on_1b:
            score += 0.01
        
        score_diff = pitch_data.get('score_diff', 0)
        if score_diff < 0:
            score += 0.03
        elif score_diff == 0:
            score += 0.02
        
        return 1 if score > self.threshold else 0


class GreedyPolicy:
    """Baseline B: Challenge whenever expected value is positive."""
    
    def __init__(self, lambda_penalty=0.02):
        self.name = "Greedy"
        self.lambda_penalty = lambda_penalty
    
    def decide(self, pitch_data, challenges_left):
        if challenges_left <= 0:
            return 0
        
        distance = pitch_data.get('dist_from_zone', 0.25)
        if pd.isna(distance):
            distance = 0.25
        
        dist_inches = distance * 12
        
        if dist_inches < 0.5:
            p_overturn = 0.50
        elif dist_inches < 1.0:
            p_overturn = 0.35
        elif dist_inches < 1.5:
            p_overturn = 0.25
        elif dist_inches < 2.0:
            p_overturn = 0.18
        elif dist_inches < 3.0:
            p_overturn = 0.10
        else:
            p_overturn = 0.05
        
        outs = pitch_data.get('outs_when_up', 0)
        strikes = pitch_data.get('strikes', 0)
        balls = pitch_data.get('balls', 0)
        
        if strikes == 2:
            if balls == 3:
                delta_re24 = 0.12
            else:
                delta_re24 = 0.08
        elif strikes == 1 and balls >= 2:
            delta_re24 = 0.06
        elif outs == 2:
            delta_re24 = 0.05
        else:
            delta_re24 = 0.04
        
        on_1b = pd.notna(pitch_data.get('on_1b')) and pitch_data.get('on_1b') != 0
        on_2b = pd.notna(pitch_data.get('on_2b')) and pitch_data.get('on_2b') != 0
        on_3b = pd.notna(pitch_data.get('on_3b')) and pitch_data.get('on_3b') != 0
        
        if on_3b:
            delta_re24 *= 1.6
        if on_2b:
            delta_re24 *= 1.4
        if on_1b and not on_2b and not on_3b:
            delta_re24 *= 1.1
        
        inning = pitch_data.get('inning', 5)
        if inning >= 9:
            delta_re24 *= 1.2
        elif inning >= 7:
            delta_re24 *= 1.1
        
        expected_value = p_overturn * delta_re24 - self.lambda_penalty * (1 - p_overturn)
        
        return 1 if expected_value > 0 else 0


class QLearningPolicy:
    """Wrapper to use trained Q-learner or Q-table as a policy."""
    name = "Q-Learning"
    
    def __init__(self, qlearner_or_qtable, encoder):
        self.encoder = encoder
        # Handle both QLearner object (training) and raw Q-table array (inference)
        if hasattr(qlearner_or_qtable, 'get_action'):
            self.qlearner = qlearner_or_qtable
            self.q_table = None
        else:
            self.qlearner = None
            self.q_table = qlearner_or_qtable
    
    def decide(self, pitch_data, challenges_left):
        if challenges_left <= 0:
            return 0
        state = self.encoder.encode(pitch_data, challenges_left)
        
        if self.qlearner is not None:
            return self.qlearner.get_action(state)
        else:
            if state >= len(self.q_table):
                return 0
            return int(np.argmax(self.q_table[state]))


# GAME SIMULATOR


class GameSimulator:
    """Simulate games for training and evaluation."""
    
    def __init__(self, data, encoder):
        self.data = data
        self.encoder = encoder
        
        self.games = {}
        for game_pk, group in data.groupby('game_pk'):
            if 'at_bat_number' in group.columns:
                sorted_group = group.sort_values(['at_bat_number', 'pitch_number'] if 'pitch_number' in group.columns else 'at_bat_number')
            else:
                sorted_group = group
            self.games[game_pk] = sorted_group
        
        self.game_ids = list(self.games.keys())
        print(f"Loaded {len(self.game_ids)} games for simulation")
    
    def simulate_episode(self, game_pk, learner, training=True):
        if game_pk not in self.games:
            return 0, 0, 0, []
        
        game_pitches = self.games[game_pk]
        
        challenges_left = 2
        total_reward = 0.0
        challenges_made = 0
        successful = 0
        decisions = []
        
        prev_state = None
        
        for idx, pitch in game_pitches.iterrows():
            state = self.encoder.encode(pitch, challenges_left)
            
            if training and prev_state is not None and isinstance(learner, QLearner):
                learner.query(state, 0)
            
            if isinstance(learner, QLearner):
                if training:
                    if prev_state is None:
                        action = learner.querysetstate(state)
                    else:
                        action = learner.a
                else:
                    action = learner.get_action(state)
            else:
                action = learner.decide(pitch, challenges_left)
            
            reward = 0.0
            if action == 1 and challenges_left > 0:
                challenges_made += 1
                is_incorrect = pitch.get('call_incorrect', False)
                
                if is_incorrect:
                    reward = abs(pitch.get('delta_re24', 0))
                    successful += 1
                else:
                    reward = -0.01
                    challenges_left -= 1
                
                total_reward += reward
                
                if training and isinstance(learner, QLearner):
                    next_state = self.encoder.encode(pitch, challenges_left)
                    learner.query(next_state, reward)
            
            decisions.append((state, action, reward))
            prev_state = state
        
        return total_reward, challenges_made, successful, decisions


# TRAINING


def train_qlearner(train_data, encoder, config):
    print(f"\nTRAINING Q-LEARNER")
    
    epochs = config.get('epochs', 100)
    alpha = config.get('alpha', 0.2)
    gamma = config.get('gamma', 0.9)
    rar = config.get('rar', 0.5)
    radr = config.get('radr', 0.995)
    dyna = config.get('dyna', 50)
    
    print(f"Epochs: {epochs}, Alpha: {alpha}, Gamma: {gamma}")
    print(f"Exploration: {rar} -> decays by {radr}, Dyna: {dyna}")
    
    learner = QLearner(
        num_states=encoder.num_states,
        num_actions=2,
        alpha=alpha,
        gamma=gamma,
        rar=rar,
        radr=radr,
        dyna=dyna
    )
    
    simulator = GameSimulator(train_data, encoder)
    
    history = {
        'epoch': [],
        'avg_reward': [],
        'avg_challenges': [],
        'success_rate': [],
        'exploration_rate': []
    }
    
    print(f"\nTraining...")
    for epoch in range(epochs):
        game_ids = simulator.game_ids.copy()
        random.shuffle(game_ids)
        
        epoch_rewards = []
        epoch_challenges = []
        epoch_successful = []
        
        games_per_epoch = min(100, len(game_ids))
        for game_pk in game_ids[:games_per_epoch]:
            try:
                reward, challenges, successful, _ = simulator.simulate_episode(
                    game_pk, learner, training=True
                )
                epoch_rewards.append(reward)
                epoch_challenges.append(challenges)
                epoch_successful.append(successful)
            except Exception as e:
                continue
        
        avg_reward = np.mean(epoch_rewards) if epoch_rewards else 0
        avg_chal = np.mean(epoch_challenges) if epoch_challenges else 0
        total_chal = sum(epoch_challenges)
        total_success = sum(epoch_successful)
        success_rate = (total_success / total_chal * 100) if total_chal > 0 else 0
        
        history['epoch'].append(epoch)
        history['avg_reward'].append(avg_reward)
        history['avg_challenges'].append(avg_chal)
        history['success_rate'].append(success_rate)
        history['exploration_rate'].append(learner.rar)
        
        learner.decay_exploration()
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}: Reward={avg_reward:.4f}, "
                  f"Chal/game={avg_chal:.2f}, Success={success_rate:.1f}%")
    
    print(f"\nTraining complete! States visited: {len(learner.experience)}")
    
    return learner, history


# EVALUATION


def evaluate_policies(test_data, encoder, qlearner):
    """Evaluate policies on test set and save per-game results."""
    print(f"\nEVALUATING POLICIES")
    
    simulator = GameSimulator(test_data, encoder)
    
    policies = [
        TerminalCountPolicy(threshold=0.08),
        GreedyPolicy(lambda_penalty=0.01),
        QLearningPolicy(qlearner, encoder)
    ]
    
    results = {}
    per_game_results = []
    
    for policy in policies:
        print(f"\nEvaluating {policy.name}...")
        
        rewards = []
        challenges = []
        successes = []
        
        for game_pk in simulator.game_ids:
            try:
                reward, chal, success, _ = simulator.simulate_episode(
                    game_pk, policy, training=False
                )
                rewards.append(reward)
                challenges.append(chal)
                successes.append(success)
                
                per_game_results.append({
                    'game_pk': game_pk,
                    'policy': policy.name,
                    'reward': reward,
                    'challenges': chal,
                    'successes': success
                })
            except:
                continue
        
        total_chal = sum(challenges)
        total_success = sum(successes)
        
        results[policy.name] = {
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'avg_challenges': np.mean(challenges),
            'success_rate': (total_success / total_chal * 100) if total_chal > 0 else 0,
            'total_games': len(rewards),
            'projected_wins': np.mean(rewards) * 162 / 10,
            'per_game_rewards': rewards
        }
        
        r = results[policy.name]
        print(f"  Games: {r['total_games']}, ER/game: {r['avg_reward']:.4f}, "
              f"Success: {r['success_rate']:.1f}%, Wins: +{r['projected_wins']:.2f}")
    
    # Save per-game results
    per_game_df = pd.DataFrame(per_game_results)
    try:
        os.makedirs('../results', exist_ok=True)
        per_game_df.to_csv('../results/per_game_results.csv', index=False)
        print(f"\nSaved per_game_results.csv")
    except Exception as e:
        print(f"\nCould not save per-game results: {e}")
    
    return results


# SAVE RESULTS


def save_results(results, qlearner, history, encoder, base_path='..'):
    output_dir = f'{base_path}/results'
    models_dir = f'{base_path}/models'
    figures_dir = f'{base_path}/results/figures'
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    # Save Q-table
    with open(f'{models_dir}/q_table.pkl', 'wb') as f:
        pickle.dump({
            'q_table': qlearner.Q,
            'num_states': qlearner.num_states,
            'num_actions': qlearner.num_actions,
            'inning_config': encoder.inning_config,
            'inning_buckets': encoder.inning_setup['labels']
        }, f)
    print(f"Saved q_table.pkl")
    
    # Save policy comparison CSV
    with open(f'{output_dir}/policy_comparison.csv', 'w') as f:
        f.write("Policy,Avg_ER_Game,Std_Dev,Success_Rate,Challenges_Game,Projected_Wins\n")
        for name, r in results.items():
            f.write(f"{name},{r['avg_reward']:.4f},{r['std_reward']:.4f},"
                    f"{r['success_rate']:.1f},{r['avg_challenges']:.2f},{r['projected_wins']:.2f}\n")
    print(f"Saved policy_comparison.csv")
    
    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(f'{output_dir}/training_history.csv', index=False)
    print(f"Saved training_history.csv")
    
    # Save training progress chart (only chart from q_learning.py)
    plt.figure(figsize=(10, 6))
    plt.plot(history['epoch'], history['avg_reward'], 'b-', linewidth=2)
    plt.xlabel('Training Epoch')
    plt.ylabel('Average Reward per Game')
    plt.title('Q-Learning Training Progress')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{figures_dir}/training_progress.png', dpi=150)
    plt.close()
    print(f"Saved training_progress.png")
    
    # Save summary text
    with open(f'{output_dir}/summary.txt', 'w') as f:
        f.write("ABS CHALLENGE STRATEGY - RESULTS SUMMARY\n")
        f.write(f"\nInning Config: {encoder.inning_setup['name']}\n")
        f.write(f"State Space: {encoder.num_states:,} states\n\n")
        
        for name, r in results.items():
            f.write(f"{name}:\n")
            f.write(f"  ER/Game: {r['avg_reward']:.4f} +/- {r['std_reward']:.4f}\n")
            f.write(f"  Challenges/Game: {r['avg_challenges']:.2f}\n")
            f.write(f"  Success Rate: {r['success_rate']:.1f}%\n")
            f.write(f"  Projected Wins: +{r['projected_wins']:.2f}\n\n")
    print(f"Saved summary.txt")


# HYPERPARAMETER EXPERIMENT


def run_inning_config_experiment(train_data, test_data, configs_to_test=None):
    """Run experiment to find best inning bucket configuration."""
    print("\nHYPERPARAMETER EXPERIMENT: INNING BUCKETS")
    
    if configs_to_test is None:
        configs_to_test = ['3bucket', '3bucket_alt1', '4bucket', '4bucket_alt1', '4bucket_alt2']
    
    experiment_results = {}
    
    for config_name in configs_to_test:
        print(f"\n--- Testing: {config_name} ---")
        
        encoder = StateEncoder(use_full_state=False, inning_config=config_name)
        
        config = {
            'epochs': 100,
            'alpha': 0.2,
            'gamma': 0.9,
            'rar': 0.5,
            'radr': 0.97,
            'dyna': 50
        }
        
        qlearner, history = train_qlearner(train_data, encoder, config)
        results = evaluate_policies(test_data, encoder, qlearner)
        
        experiment_results[config_name] = {
            'config_name': encoder.inning_setup['name'],
            'num_states': encoder.num_states,
            'q_learning_er': results['Q-Learning']['avg_reward'],
            'q_learning_wins': results['Q-Learning']['projected_wins'],
            'q_learning_success': results['Q-Learning']['success_rate'],
            'improvement_vs_greedy': results['Q-Learning']['avg_reward'] - results['Greedy']['avg_reward'],
            'full_results': results
        }
    
    # Summary
    print("EXPERIMENT SUMMARY")
    
    print(f"\n{'Config':<20} {'States':>8} {'ER/Game':>10} {'Wins':>8}")
    
    best_config = None
    best_er = -1
    
    for config_name, exp in experiment_results.items():
        print(f"{config_name:<20} {exp['num_states']:>8,} {exp['q_learning_er']:>10.4f} "
              f"{exp['q_learning_wins']:>8.2f}")
        
        if exp['q_learning_er'] > best_er:
            best_er = exp['q_learning_er']
            best_config = config_name
    
    print(f"\nBEST: {best_config}")
    
    # Save experiment results
    try:
        os.makedirs('../results', exist_ok=True)
        rows = []
        for config_name, exp in experiment_results.items():
            rows.append({
                'Config': config_name,
                'States': exp['num_states'],
                'ER_Game': round(exp['q_learning_er'], 4),
                'Wins_Season': round(exp['q_learning_wins'], 2),
                'Success_Rate': round(exp['q_learning_success'], 1),
                'vs_Greedy': round(exp['improvement_vs_greedy'], 4)
            })
        pd.DataFrame(rows).to_csv('../results/hyperparameter_experiment.csv', index=False)
        print(f"\nSaved hyperparameter_experiment.csv")
    except Exception as e:
        print(f"\nCould not save experiment CSV: {e}")
    
    return experiment_results


def main():
    print("ABS CHALLENGE STRATEGY - Q-Learning")
    print("Christopher Martinez | Georgia Tech")
    print(f"Seed: {SEED}")
    
    # Load data
    print("\n[1/5] Loading data...")
    
    path_configs = [
        ('../data/processed/', '..'),
        ('data/processed/', '.'),
    ]
    
    data_path = None
    base_path = None
    for dp, bp in path_configs:
        if os.path.exists(f'{dp}train_data.csv'):
            data_path = dp
            base_path = bp
            break
    
    if data_path is None:
        print("Error: Data files not found.")
        return
    
    try:
        train_data = pd.read_csv(f'{data_path}train_data.csv')
        val_data = pd.read_csv(f'{data_path}val_data.csv')
        test_data = pd.read_csv(f'{data_path}test_data.csv')
        print(f"  Train: {len(train_data):,} pitches ({train_data['game_pk'].nunique()} games)")
        print(f"  Val:   {len(val_data):,} pitches ({val_data['game_pk'].nunique()} games)")
        print(f"  Test:  {len(test_data):,} pitches ({test_data['game_pk'].nunique()} games)")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # CHOOSE INNING CONFIGURATION HERE
    
    INNING_CONFIG = '4bucket_alt2'  # BEST based on experiments
    
    # To run hyperparameter experiment, uncomment these two lines:
    # experiment_results = run_inning_config_experiment(train_data, test_data)
    # return
    
    
    print("\n[2/5] Initializing encoder...")
    encoder = StateEncoder(use_full_state=False, inning_config=INNING_CONFIG)
    
    config = {
        'epochs': 100,
        'alpha': 0.1,
        'gamma': 0.9,
        'rar': 0.5,
        'radr': 0.97,
        'dyna': 50
    }
    
    print("\n[3/5] Training Q-learner...")
    qlearner, history = train_qlearner(train_data, encoder, config)
    
    print("\n[4/5] Evaluating policies...")
    results = evaluate_policies(test_data, encoder, qlearner)
    
    print("\n[5/5] Saving results...")
    save_results(results, qlearner, history, encoder, base_path)
    
    # Final summary
    print("COMPLETE!")
    print(f"\nConfig: {encoder.inning_setup['name']} ({encoder.num_states:,} states)")
    print("\nResults:")
    for name, r in results.items():
        print(f"  {name:15s}: {r['avg_reward']:.4f} ER/game -> +{r['projected_wins']:.2f} wins")
    
    print(f"\nNext: Run 'python statistical_analysis.py' to generate all charts.")


if __name__ == "__main__":
    main()