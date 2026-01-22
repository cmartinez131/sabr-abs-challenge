<!-- Copilot instructions for contributors and AI coding agents -->
# Copilot / AI Contributor Instructions

Purpose: Quickly orient an automated coding assistant to the structure, intent, and practical workflows of this repository so it can make safe, useful edits.

- Big picture:
  - This repo implements a small reinforcement-learning project (Q-learning) for an ABS/Minor League dataset. The training entrypoint is `main.py` which runs a training loop. Exploratory work and visualizations live in `notebooks/` and scripts at the repo root (e.g., `seaborn-charts.py`). Core components are in `src/`.

- Key files and what they do (quick lookups):
  - `main.py`: training loop entrypoint — run `python main.py` to start training.
  - `data/abs_minor_league_games.csv`: canonical dataset used by notebooks and scripts.
  - `src/environment.py`: game/environment logic (state transitions, rewards) — look here for "If I challenge, what happens?" behavior.
  - `src/q_learning_agent.py`: the agent class; contains `q_table`, `choose_action`, `learn` patterns.
  - `src/utils.py`: helper functions such as `calculate_run_expectancy(bases, outs)` — use this for reward/feature calculations.
  - `notebooks/02_train_agent.ipynb`: canonical training notebook showing hyperparameters and an alternate training flow — consult for examples when adding or refactoring training code.
  - `output/figures/` and `output/model_checkpoints/`: expected locations for visualization output and model artifacts. Keep these paths stable when modifying save/load behavior.

- Project-specific conventions and patterns:
  - Place core logic under `src/`. Scripts at the repository root are for orchestration or plotting only.
  - Persisted outputs and models go under `output/` (figures or model_checkpoints). Avoid hardcoding other output paths.
  - Utilities that compute domain-specific values (e.g., run expectancy) are centralized in `src/utils.py` — prefer reusing these helpers.
  - Notebooks are treated as reference implementations and canonical examples of usage (do not rewrite notebook logic without ensuring parity in `src/`).

- Development workflows (what an agent should run):
  - Start training locally: `python main.py` (this is the primary run command). If the user asks to reproduce notebook results, open the corresponding notebook and run cells interactively.
  - Inspect data: open `data/abs_minor_league_games.csv` or run the notebooks; there is no `requirements.txt` in the repo — install a minimal data-science stack if needed (`pandas, numpy, matplotlib, seaborn`).

- Editing guidance for AI agents (do this, not that):
  - Do: Make focused, minimal changes that preserve the `src/` / `notebooks/` separation and output paths under `output/`.
  - Do: Reference `src/utils.py` functions (e.g., `calculate_run_expectancy`) instead of reimplementing domain logic.
  - Do: When changing training behavior, update `notebooks/02_train_agent.ipynb` or add a short runnable script demonstrating usage.
  - Don't: Move core logic out of `src/` into notebooks or root scripts. Don't change output directories without updating `main.py` and notebooks consistently.

- Quick examples (where to look for patterns):
  - Reward/feature code: see `src/utils.py` -> `calculate_run_expectancy(bases, outs)`.
  - Agent API shape: see `src/q_learning_agent.py` for `q_table`, `choose_action`, and `learn` method signatures.

- If you need clarification from a human:
  - Ask where model checkpoint format should be standardized (pickle vs numpy vs torch).
  - Ask whether notebooks should be converted into deterministic scripts for CI.

If any section is unclear or you'd like the agents to follow stricter conventions (formatting, CI, or dependency pinning), tell me which area to expand and I'll iterate.
