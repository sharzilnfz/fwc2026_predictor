# 🏆 World Cup 2026 Predictor — Learn-By-Doing Capstone Guide

This is your complete learning roadmap. You build everything yourself. Each phase has **lessons** (what to learn), **tasks** (what to code), **checkpoints** (how to verify), and **reference** (what to consult). I've set up the project workspace using the [/teach](file:///Users/sharzilnafis/Desktop/Project/skills/skills/productivity/teach/SKILL.md) skill framework.

> [!IMPORTANT]
> **The 2026 World Cup started June 11.** The tournament runs until July 19. This creates a unique portfolio opportunity: make predictions, then compare them to reality. Prioritize getting a working baseline ASAP.

---

## Your Learning Workspace

I've created the full project at `~/Desktop/Project/worldcup-predictor/` with these teaching documents:

| File                                                                                       | Purpose                                                                         |
| ------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------- |
| [MISSION.md](file:///Users/sharzilnafis/Desktop/Project/worldcup-predictor/MISSION.md)     | Why you're doing this — grounds every decision                                  |
| [RESOURCES.md](file:///Users/sharzilnafis/Desktop/Project/worldcup-predictor/RESOURCES.md) | Curated, trusted sources — datasets, courses, papers, communities               |
| [GLOSSARY.md](file:///Users/sharzilnafis/Desktop/Project/worldcup-predictor/GLOSSARY.md)   | Shared language — 30+ terms you'll use daily (ELO, K-factor, Brier score, etc.) |
| [NOTES.md](file:///Users/sharzilnafis/Desktop/Project/worldcup-predictor/NOTES.md)         | Teaching notes & learner preferences                                            |

### Project Structure (already scaffolded)

```
worldcup-predictor/
├── MISSION.md              # Your learning mission
├── RESOURCES.md             # Curated learning resources
├── GLOSSARY.md              # Project terminology
├── NOTES.md                 # Teaching notes
├── lessons/                 # HTML lesson files (created as you learn)
├── learning-records/        # Evidence of what you've mastered
├── reference/               # Cheat sheets & quick-reference docs
├── data/
│   ├── raw/                 # Downloaded CSVs (gitignored)
│   ├── processed/           # Cleaned, feature-engineered data
│   └── predictions/         # Model outputs
├── notebooks/               # Jupyter notebooks (your main workspace)
├── src/                     # Reusable Python modules
├── dashboard/               # Streamlit app
│   ├── pages/
│   └── assets/
├── models/                  # Saved model artifacts
└── tests/                   # Unit tests
```

---

## Phase 0: Environment Setup

**⏱ Time: 1-2 hours**

### Tasks

1. **Set up the Python environment**

   ```bash
   cd ~/Desktop/Project/worldcup-predictor
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Create `requirements.txt`** with these dependencies:

   ```
   pandas>=2.0
   numpy>=1.24
   scikit-learn>=1.3
   xgboost>=2.0
   lightgbm>=4.0
   torch>=2.0
   matplotlib>=3.7
   seaborn>=0.12
   plotly>=5.15
   shap>=0.42
   optuna>=3.3
   streamlit>=1.28
   jupyter>=1.0
   joblib>=1.3
   ```

3. **Install**: `pip install -r requirements.txt`

4. **Download datasets** from Kaggle:
   - Go to [Kaggle](https://www.kaggle.com/) → create free account → download:
     - "International football results from 1872 to 2026" → save to `data/raw/`
     - "FIFA World Cup 1930–2026: Ultimate ML Dataset" → save to `data/raw/`
     - FIFA rankings dataset → save to `data/raw/`
   - Or use the Kaggle API:
     ```bash
     pip install kaggle
     # Set up ~/.kaggle/kaggle.json with your API key
     kaggle datasets download -d martj42/international-football-results-from-1872-to-2017 -p data/raw/ --unzip
     ```

5. **Initialize git**:
   ```bash
   git init
   echo "venv/\ndata/raw/\nmodels/*.pt\nmodels/*.joblib\n__pycache__/\n.ipynb_checkpoints/" > .gitignore
   git add .
   git commit -m "Initial project scaffold"
   ```

### Checkpoint

- [ ] `python -c "import pandas, sklearn, xgboost, torch; print('All good')"` runs without errors
- [ ] `data/raw/` contains at least `results.csv` with 40,000+ rows
- [ ] Git repo initialized with first commit

---

## Phase 1: Data Exploration & Cleaning

**⏱ Time: 4-6 hours | 📓 Notebook: `01_data_exploration.ipynb`**

### What You'll Learn

- How to explore and understand a messy real-world dataset
- Common data quality issues (missing values, inconsistent naming, duplicates)
- How to visualize distributions and trends in football data

### Tasks

#### 1.1 Load and Inspect

Create `notebooks/01_data_exploration.ipynb` and explore:

```python
# Your starting point — fill in the rest yourself
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('../data/raw/results.csv')

# TODO: What are the columns? What do the first 10 rows look like?
# TODO: How many rows? How many unique teams? Date range?
# TODO: What does the 'tournament' column contain? How many World Cup matches?
# TODO: Any missing values? Which columns?
# TODO: What does the 'neutral' column mean?
```

#### 1.2 Answer These Questions (Write Your Findings In The Notebook)

1. How many total matches are in the dataset? How many are World Cup matches?
2. What's the distribution of home wins vs away wins vs draws? (Plot a bar chart)
3. How has the average number of goals per match changed over time? (Plot a line chart by decade)
4. Which teams have the most World Cup matches? (Top 20 bar chart)
5. Is there a significant home advantage? What's the win rate for home vs away teams?
6. Are there any team name inconsistencies? (e.g., "West Germany" vs "Germany")

#### 1.3 Clean the Data

- Normalize team names (handle historical name changes)
- Parse dates into proper datetime objects
- Filter to post-2000 matches (more relevant for modern football)
- Save cleaned data to `data/processed/matches_clean.csv`

### Key Concepts to Understand

> Read the [GLOSSARY](file:///Users/sharzilnafis/Desktop/Project/worldcup-predictor/GLOSSARY.md) entries for: **feature**, **target**, **training set**, **validation set**

### Checkpoint

- [ ] Notebook runs end-to-end without errors
- [ ] At least 5 visualizations that tell a story about the data
- [ ] Cleaned CSV saved with no missing values in key columns
- [ ] You can explain the home advantage effect in one sentence

### 💡 Skill Tip

> If you're unsure about any analysis step, use `/teach` and ask me to create a lesson on that specific concept. If you want to stress-test your data cleaning plan, use `/grill-me`.

---

## Phase 2: Feature Engineering

**⏱ Time: 8-12 hours | 📓 Notebook: `02_feature_engineering.ipynb` | 📦 Module: `src/elo.py`, `src/features.py`**

This is the **highest-impact phase**. Research shows feature engineering matters more than model choice in football prediction.

### What You'll Learn

- How ELO ratings work and how to implement them from scratch
- How to compute rolling/windowed statistics
- How to think about what information is available BEFORE a match happens (no data leakage!)

### Tasks

#### 2.1 Build the ELO Rating System (`src/elo.py`)

This is your first standalone module. Build it with tests ([/tdd](file:///Users/sharzilnafis/Desktop/Project/skills/skills/engineering/tdd/SKILL.md)).

**How ELO works** (implement this yourself):

1. Every team starts with ELO = 1500
2. Before a match, calculate expected outcome: `E_A = 1 / (1 + 10^((R_B - R_A) / 400))`
3. After the match, update: `R_A_new = R_A + K * (S_A - E_A)`
   - `S_A` = 1 (win), 0.5 (draw), 0 (loss)
   - `K` = weight factor (how much each match matters)
4. Use different K-factors by tournament importance:
   - World Cup: K=60
   - Continental championship (Copa America, Euros): K=50
   - World Cup qualifiers: K=40
   - Friendlies: K=20

**Your module should**:

- Take the full match history and compute ELO for every team at every point in time
- Return a function: `get_elo(team, date) → float`
- Handle teams that don't exist yet (return 1500)

**Write tests first** (`tests/test_elo.py`):

```python
# Test ideas — implement these yourself
# 1. New team starts at 1500
# 2. Winner's ELO goes up, loser's goes down
# 3. Bigger upset = bigger ELO change
# 4. Draw between equal teams = no change
# 5. K-factor for World Cup > K-factor for friendly
```

> [!TIP]
> Use `/prototype` to build a quick throwaway version first. Test it on 10 matches manually. Does it make sense? Then formalize into a module with `/tdd`.

#### 2.2 Build the Feature Pipeline (`src/features.py`)

For each match, compute these features **using only data available BEFORE the match**:

| Feature               | How to Compute                                 | Priority        |
| --------------------- | ---------------------------------------------- | --------------- |
| `elo_home`            | ELO rating of home team on match date          | 🔴 Critical     |
| `elo_away`            | ELO rating of away team on match date          | 🔴 Critical     |
| `elo_diff`            | `elo_home - elo_away`                          | 🔴 Critical     |
| `rank_home`           | FIFA ranking of home team                      | 🟡 High         |
| `rank_away`           | FIFA ranking of away team                      | 🟡 High         |
| `rank_diff`           | `rank_home - rank_away`                        | 🟡 High         |
| `form_home`           | Win rate over team's last 10 matches           | 🟡 High         |
| `form_away`           | Win rate over team's last 10 matches           | 🟡 High         |
| `goals_scored_home`   | Avg goals scored in last 10 matches            | 🟡 High         |
| `goals_conceded_home` | Avg goals conceded in last 10 matches          | 🟡 High         |
| `goals_scored_away`   | Avg goals scored in last 10 matches            | 🟡 High         |
| `goals_conceded_away` | Avg goals conceded in last 10 matches          | 🟡 High         |
| `gd_home`             | Avg goal difference in last 10 matches         | 🟡 High         |
| `gd_away`             | Avg goal difference in last 10 matches         | 🟡 High         |
| `h2h_wins_home`       | Historical wins for home team vs this opponent | 🟢 Medium       |
| `h2h_matches`         | Total historical meetings between these teams  | 🟢 Medium       |
| `is_neutral`          | Boolean: neutral venue?                        | 🟢 Medium       |
| `wc_appearances_home` | Team's total World Cup appearances             | 🟢 Medium       |
| `wc_appearances_away` | Team's total World Cup appearances             | 🟢 Medium       |
| `confederation_home`  | One-hot encoded confederation                  | 🔵 Nice-to-have |
| `confederation_away`  | One-hot encoded confederation                  | 🔵 Nice-to-have |

> [!WARNING]
> **Data leakage is the #1 beginner mistake.** Never use information from the future. When computing rolling stats for a match on June 15, you can only use matches BEFORE June 15. Double-check this in every feature.

#### 2.3 Create the Training Dataset

- Apply the feature pipeline to all matches
- Create the target variable: `outcome` = 0 (away win), 1 (draw), 2 (home win)
- Split into: training (pre-2018), validation (2018-2021), test (2022 World Cup)
- Save to `data/processed/features_train.csv`, `features_val.csv`, `features_test.csv`

### Checkpoint

- [ ] `pytest tests/test_elo.py` — all tests pass
- [ ] ELO ratings look sensible: Brazil, Germany, France should be top-rated historically
- [ ] Feature matrix has no NaN values (or they're handled)
- [ ] No data leakage: features for a match only use prior data
- [ ] Train/val/test split is temporal (no future data in training)

### 💡 Skill Tips

> - Use `/grill-me` before starting: "Grill me on my feature engineering plan — am I missing anything? Is there data leakage?"
> - Use `/tdd` for `src/features.py` — write a test that asserts rolling win rate for a known team sequence
> - Consult the [ELO paper](https://ideas.repec.org/p/nms/nmsrif/3110.html) from [RESOURCES.md](file:///Users/sharzilnafis/Desktop/Project/worldcup-predictor/RESOURCES.md)

---

## Phase 3: Model Building

**⏱ Time: 10-15 hours | 📓 Notebooks: `03_model_training.ipynb`, `04_model_evaluation.ipynb`**

### What You'll Learn

- How to build and evaluate classification models progressively (simple → complex)
- Hyperparameter tuning with Optuna
- Building a neural network in PyTorch from scratch
- Model comparison and ensemble techniques

### Model Progression (Build In This Order)

#### 3.1 Model 1: Logistic Regression (Baseline) — _Start Here_

```python
# Skeleton — implement yourself in 03_model_training.ipynb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# TODO: Load features_train.csv and features_val.csv
# TODO: Create a pipeline with StandardScaler + LogisticRegression
# TODO: Fit on training data
# TODO: Predict on validation data
# TODO: Print accuracy, classification report, confusion matrix
# TODO: What accuracy would random guessing give? (hint: class distribution)
```

**Questions to answer**:

- What's the baseline accuracy? (Predicting the most common class every time)
- Does the logistic regression beat random guessing? By how much?
- What are the coefficients? Which features matter most?

#### 3.2 Model 2: XGBoost (Primary Model)

```python
# Skeleton — implement yourself
import xgboost as xgb
from sklearn.model_selection import cross_val_score

# TODO: Train an XGBoost classifier with default params
# TODO: Compare accuracy to logistic regression — is it better?
# TODO: Plot feature importance (xgb.plot_importance)
# TODO: Use Optuna to tune: max_depth, learning_rate, n_estimators, subsample
# TODO: Retrain with best params, evaluate on validation set
```

**Optuna tuning template** (adapt to your code):

```python
import optuna

def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
    }
    # TODO: Train model with params, return validation accuracy
    pass

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
```

#### 3.3 Model 3: Poisson Regression (Goal Prediction)

This is a different approach: instead of predicting win/draw/loss directly, predict the number of goals each team scores. Then derive outcome probabilities.

```python
# Skeleton — research and implement yourself
from sklearn.linear_model import PoissonRegressor

# TODO: Train TWO models:
#   - Model A: predict home_goals given features
#   - Model B: predict away_goals given features
# TODO: For a match, predict λ_home and λ_away (expected goals)
# TODO: Use scipy.stats.poisson to compute P(home_goals=i, away_goals=j) for i,j in 0..5
# TODO: Sum probabilities: P(home win) = sum P(home>away), P(draw) = sum P(home==away), etc.
# TODO: Compare these derived probabilities to XGBoost's probabilities
```

> [!TIP]
> The Poisson approach is portfolio gold — it shows you understand different modeling paradigms, not just "throw XGBoost at everything."

#### 3.4 Model 4: PyTorch Neural Network (Deep Learning)

Build a feed-forward neural network from scratch:

```python
# Skeleton — implement yourself
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class MatchPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_classes=3):
        super().__init__()
        # TODO: Build layers with BatchNorm and Dropout
        # Suggested architecture: input → 128 → 64 → 32 → 3
        pass

    def forward(self, x):
        # TODO: Implement forward pass
        pass

# TODO: Create DataLoader from your feature matrices
# TODO: Write training loop (loss function: CrossEntropyLoss)
# TODO: Track training and validation loss per epoch
# TODO: Plot loss curves — are you overfitting?
# TODO: Add early stopping
# TODO: Save the best model to models/nn_model.pt
```

**Key decisions to make** (use `/grill-me`):

- How many hidden layers? How wide?
- What dropout rate? (start with 0.3)
- Learning rate? (start with 0.001, try a learning rate finder)
- Batch size? (start with 64)
- When to stop training?

#### 3.5 Ensemble

```python
# Skeleton — implement yourself
# TODO: Get probability predictions from XGBoost, Poisson, and Neural Net
# TODO: Simple average: P_ensemble = (P_xgb + P_poisson + P_nn) / 3
# TODO: Weighted average: optimize weights on validation set
# TODO: Compare ensemble accuracy to individual models
```

### Checkpoint (use `04_model_evaluation.ipynb`)

- [ ] All 4 models trained and saved
- [ ] Comparison table:

| Model               | Val Accuracy | Val F1 (macro) | Val Log-Loss |
| ------------------- | ------------ | -------------- | ------------ |
| Random baseline     | ~33%         | -              | -            |
| Logistic Regression | ?            | ?              | ?            |
| XGBoost             | ?            | ?              | ?            |
| Poisson (derived)   | ?            | ?              | ?            |
| Neural Network      | ?            | ?              | ?            |
| Ensemble            | ?            | ?              | ?            |

- [ ] Feature importance plot (SHAP or XGBoost built-in)
- [ ] Learning curves for the neural network (train vs val loss)
- [ ] The ensemble beats every individual model (or you understand why not)

### 💡 Skill Tips

> - Start with logistic regression. Don't touch XGBoost until LR is working end-to-end. ([Karpathy Guidelines](file:///Users/sharzilnafis/Desktop/Project/skills/karpathy%20skill.md): Simplicity First)
> - If a model does something weird, use `/diagnose` — systematic debugging, not guessing
> - Use `/prototype` to quickly test "does my PyTorch training loop converge on a toy problem?"

---

## Phase 4: Backtest on 2022 World Cup

**⏱ Time: 3-4 hours | 📓 Notebook: `04_model_evaluation.ipynb` (continued)**

### What You'll Learn

- How to validate a model on truly unseen data
- How to communicate model performance honestly
- The difference between accuracy and calibration

### Tasks

1. **Predict every 2022 World Cup match** using your ensemble
2. **Compare predictions to actual results** — match by match
3. **Compute metrics**: accuracy, Brier score, calibration curve
4. **Create a compelling visualization**: predicted vs actual bracket
5. **Write an honest assessment**: where did the model succeed? Where did it fail? (Upsets like Saudi Arabia beating Argentina are EXPECTED to be wrong — that's what makes them upsets)

### Checkpoint

- [ ] Predictions for all 64 matches of 2022 World Cup
- [ ] Accuracy above 45% on 3-class prediction (competitive with research benchmarks)
- [ ] Calibration plot showing predicted probability vs actual frequency
- [ ] Honest writeup of strengths and weaknesses

---

## Phase 5: 2026 World Cup Predictions

**⏱ Time: 4-6 hours | 📓 Notebook: `05_tournament_simulation.ipynb` | 📦 Module: `src/simulator.py`**

### What You'll Learn

- Monte Carlo simulation for tournament prediction
- How to handle group stage tiebreakers and knockout draws
- How to communicate uncertainty (probability distributions, not point predictions)

### Tasks

#### 5.1 Build the Match Predictor (`src/predictor.py`)

```python
# TODO: Function that takes (team_a, team_b, date) and returns:
#   - P(team_a wins), P(draw), P(team_b wins)
#   - Predicted scoreline (from Poisson model)
```

#### 5.2 Build the Tournament Simulator (`src/simulator.py`)

```python
# TODO: Implement the 2026 World Cup structure:
#   - 12 groups of 4
#   - Top 2 + 8 best 3rd place → Round of 32
#   - Then: R16, QF, SF, 3rd place, Final
#   - No draws in knockout (use win probabilities only, renormalized)

# TODO: Monte Carlo simulation (run 10,000 times):
#   1. For each group match, sample outcome from model probabilities
#   2. Resolve group standings (points, GD, H2H tiebreakers)
#   3. Simulate knockout bracket
#   4. Record: who won, who reached the final, who reached each stage

# TODO: Output:
#   - Win probability for each team (bar chart)
#   - Most likely final matchup
#   - Probability of reaching each stage for each team
```

#### 5.3 Generate Predictions

- Predict every group stage match
- Run Monte Carlo simulation
- Save predictions to `data/predictions/`

### Checkpoint

- [ ] Match-by-match predictions for all 72 group stage matches
- [ ] Monte Carlo simulation runs 10,000 iterations
- [ ] Win probability chart for all 48 teams
- [ ] Predicted group standings with probabilities
- [ ] Predicted knockout bracket

---

## Phase 6: Streamlit Dashboard

**⏱ Time: 8-12 hours | 📦 `dashboard/app.py`**

### What You'll Learn

- Building interactive data apps with Streamlit
- Data visualization with Plotly
- UI/UX for data science projects

### Pages to Build

1. **🎯 Match Predictions** — dropdown to select any match, shows probabilities + predicted score
2. **🏟️ Tournament Bracket** — interactive bracket with probability coloring
3. **📊 Team Analysis** — select a team, see ELO history, form, path to final
4. **🧠 Model Insights** — SHAP values, model comparison, 2022 backtest results

### Design Requirements

- Dark theme
- Team flag icons (find free flag icon sets online)
- Animated probability bars
- Mobile-responsive

### Checkpoint

- [ ] `streamlit run dashboard/app.py` launches without errors
- [ ] All 4 pages functional and visually polished
- [ ] Can select any match and see predictions
- [ ] Can simulate the tournament with a button click

---

## Phase 7: Documentation & Deployment

**⏱ Time: 4-6 hours**

### Tasks

1. **README.md** — Portfolio quality:
   - Project title + one-line description
   - Screenshots/GIFs of the dashboard
   - Methodology section with architecture diagram
   - Results summary with key metrics
   - How to run locally
   - Link to live demo
   - Tech stack badges
   - License

2. **Deploy to Streamlit Cloud** (free):

   ```bash
   # Push to GitHub, then connect via streamlit.io/cloud
   ```

3. **Write a blog post / LinkedIn article** about your approach (optional but high-impact)

### Checkpoint

- [ ] README has screenshots and clear methodology
- [ ] Live demo URL works
- [ ] Someone unfamiliar with ML can understand what the project does from the README

---

## Recommended Learning Order & Timeline

Given the World Cup is live NOW, here's the prioritized sequence:

```
Week 1 (June 12-18): Phases 0 + 1 + 2 (Setup, Data, Features)
  → Get data loaded, ELO computed, features ready
  → You can start comparing your ELO rankings to FIFA rankings

Week 2 (June 19-25): Phase 3 (Models)
  → Train all 4 models + ensemble
  → You now have a working predictor

Week 3 (June 26 - July 2): Phases 4 + 5 (Backtest + 2026 Predictions)
  → Validate on 2022, generate 2026 predictions
  → Compare your group stage predictions to actual results!

Week 4 (July 3-9): Phase 6 (Dashboard)
  → Build the Streamlit app
  → Update predictions as knockout rounds begin

Week 5 (July 10-19): Phase 7 (Polish & Deploy)
  → Documentation, deployment, blog post
  → Final predictions for SF/Final — compare to reality!
```

> [!IMPORTANT]
> **Don't try to be perfect before moving forward.** Get a baseline logistic regression working on day 3-4, make your first predictions, and THEN iterate. A working ugly model is infinitely more valuable than a planned beautiful model.

---

## How to Ask Me for Help

When you're stuck, come back to me with context. Here are the best ways to use our sessions:

| What You Need   | What to Say                                                         |
| --------------- | ------------------------------------------------------------------- |
| Learn a concept | "Teach me about ELO ratings" (or use `/teach`)                      |
| Design review   | "Grill me on my feature engineering plan" (or use `/grill-me`)      |
| Debug an issue  | "My model predicts draws with 0% probability" (or use `/diagnose`)  |
| Understand code | "Zoom out on my features.py module" (or use `/zoom-out`)            |
| Quick prototype | "Help me prototype my Monte Carlo simulation" (or use `/prototype`) |
| Break down work | "Convert Phase 3 into trackable issues" (or use `/to-issues`)       |

---

## Open Questions For You

> [!IMPORTANT]
> **PyTorch vs TensorFlow**: I've recommended PyTorch (industry standard for research, increasingly for production). Are you comfortable with this, or prefer TensorFlow/Keras?

> [!IMPORTANT]
> **Speed vs Depth**: Do you want to speed-run to get predictions out ASAP (skip Poisson and NN, just do LR + XGBoost), or take the full learning path? The tournament is happening NOW — there's a real trade-off.

> [!IMPORTANT]
> **Data Download**: Do you have a Kaggle account, or should I help you find alternative download methods for the datasets?
