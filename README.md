# ABS Challenge Strategy Optimization

A reinforcement learning approach to optimizing challenge decisions under MLB's Automated Ball-Strike (ABS) system. This research is for my presentation submission at SABR Analytics Conference 2026.

**Author:** Christopher Martinez, Georgia Institute of Technology

## The Problem

MLB's ABS Challenge System gives teams 2 opportunities per game to challenge ball/strike calls. A successful challenge is retained; an unsuccessful one is lost. This creates a sequential decision problem: when should a team use their limited challenges to maximize expected runs?

## Approach

I framed this as a Markov Decision Process and solved it with tabular Q-learning. The key insight is that a greedy policy (challenge whenever EV > 0) ignores the **option value** of saving challenges for high-leverage situations.

**State Space:** ~2,500 discrete states encoding inning, outs, count, runners, score differential, challenges remaining, and pitch distance from zone edge.

**Reward Function:** Delta RE24 (change in run expectancy) for successful challenges, small penalty for unsuccessful ones.

## Results

| Policy | ER/Game | Projected Wins/Season |
|--------|---------|----------------------|
| Terminal-Count Heuristic | ~0.06 | +0.9 |
| Greedy (EV-based) | ~0.08 | +1.3 |
| **Q-Learning** | **~0.12** | **+1.89** |

The Q-learning agent outperforms baselines by learning to save challenges for late-inning, high-leverage situations rather than using them on any positive-EV opportunity.

## Project Structure

```
src/
    - data_collection.py        # Fetch MLB Statcast data
    - data_preprocessing.py     # Process pitches, compute RE24
    - q_learning.py             # Train model, save CSVs
    - statistical_analysis.py   # Generate ALL presentation charts and statistical analysis
data/
    - raw/                      # Raw Statcast pitch data
    - processed/                # Engineered features for training
models/                       # Trained Q-tables
results/                      # Figures and comparison tables
```

## Piepline

```bash
# 1. Collect data (requires pybaseball)
python src/data_collection.py

# 2. Preprocess and engineer features
python src/data_preprocessing.py

# 3. Train Q-learner and evaluate policies
python src/q_learning.py

# 4. Generate statistical analysis figures
python src/statistical_analysis.py
```

## Dependencies

- Python 3.8+
- pybaseball
- pandas, numpy
- scikit-learn
- matplotlib, scipy


## Installation
```bash
pip install -r requirements.txt
```

## Data

Training data comes from MLB Statcast (June-September 2024) accessed via the pybaseball library. Since the ABS challenge system wasn't fully deployed in MLB during this period, I simulate challenge outcomes using pitch location as ground truth. A pitch inside the zone is a strike, outside is a ball. The P(overturn) is modeled based on distance from the zone edge.

## Key Technical Decisions

1. **Split by game, not pitch** to prevent within-game correlation leakage
2. **Tabular Q-learning over deep RL** because the state space is interpretable matters for a baseball audience
3. **Dyna model-based updates** to improve sample efficiency with limited game data
4. **Inning bucketing** (1-4, 5-7, 8-9, 10+)

## Limitations

- Simulated challenge outcomes, not real ABS data
- No pitcher/batter-specific features
- Assumes teams start with 2 challenges 
