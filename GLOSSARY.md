# World Cup Predictor Glossary

Machine learning and football analytics terminology used throughout this project.

## ML/DL Core

**Feature**:
A measurable property of an observation (match) used as model input — e.g., ELO difference, rolling win rate.
_Avoid_: variable, column (when referring to the conceptual unit)

**Target**:
The variable the model is trying to predict — match outcome (win/draw/loss) or goal count.
_Avoid_: label (acceptable in classification context), y-variable

**Training set**:
The subset of historical matches used to fit model parameters.
_Avoid_: training data (acceptable informally)

**Validation set**:
The held-out subset used to tune hyperparameters and estimate generalization. In this project: the 2022 World Cup matches.
_Avoid_: dev set, test set (reserved for truly unseen data)

**Overfitting**:
When a model memorizes training data noise instead of learning generalizable patterns — shows high training accuracy but poor validation accuracy.
_Avoid_: overlearning

**Hyperparameter**:
A setting chosen before training that controls model behavior (learning rate, tree depth, K-factor) — not learned from data.
_Avoid_: parameter (reserved for values learned during training, like weights)

**Ensemble**:
Combining predictions from multiple models to improve accuracy and reduce variance.
_Avoid_: model blending (acceptable as a specific technique name)

**Logistic regression**:
A linear model that predicts class probabilities via a sigmoid function. Used as the baseline model.
_Avoid_: logit model

**Gradient boosting**:
An ensemble technique that trains many weak learners (trees) sequentially, each correcting the errors of the previous. XGBoost and LightGBM are implementations.
_Avoid_: boosted trees (informal, acceptable)

**Neural network**:
A model composed of layers of interconnected nodes (neurons) that learn non-linear transformations. The deep learning component.
_Avoid_: deep learning model (acceptable informally)

**Loss function**:
The mathematical function measuring how wrong the model's predictions are. Training minimizes this.
_Avoid_: cost function (acceptable synonym), error function

**Cross-entropy loss**:
The standard loss function for classification — penalizes confident wrong predictions heavily.
_Avoid_: log loss (acceptable synonym, used in evaluation context)

## Football Analytics

**ELO rating**:
A dynamic numerical rating of team strength, updated after each match based on expected vs actual outcome. Higher = stronger team.
_Avoid_: ELO score, power rating

**K-factor**:
The ELO hyperparameter controlling how much a single match result shifts the rating. Higher K = more reactive to recent results. In this project: K=40 (World Cup), K=20 (qualifiers), K=10 (friendlies).
_Avoid_: sensitivity

**Rolling statistic**:
A feature computed over a sliding window of a team's last N matches (e.g., rolling win rate over last 10 games).
_Avoid_: moving average (acceptable for the specific computation)

**Head-to-head record**:
The historical win/draw/loss counts between two specific teams across all their previous meetings.
_Avoid_: H2H, matchup history

**Confederation**:
One of six continental governing bodies (AFC, CAF, CONCACAF, CONMEBOL, OFC, UEFA) that organizes regional football. Used as a categorical feature.
_Avoid_: region, zone

**Neutral venue**:
A match location where neither team has home advantage. All 2026 World Cup matches are neutral (except semi-home for USA/MEX/CAN).
_Avoid_: neutral ground

## Evaluation

**Accuracy**:
Fraction of matches where the predicted outcome class matches the actual outcome. Simple but misleading when classes are imbalanced.
_Avoid_: hit rate

**Brier score**:
Measures the calibration of predicted probabilities — lower is better. Critical for Monte Carlo simulation: if probabilities are off, simulation outputs are meaningless.
_Avoid_: probability score

**F1-score (macro)**:
The harmonic mean of precision and recall, averaged equally across all classes. Fairer than accuracy when class sizes differ (wins vs draws).
_Avoid_: F-measure

**Calibration**:
How well predicted probabilities match observed frequencies — "when I say 70% win, does the team actually win 70% of the time?"
_Avoid_: reliability

## Simulation

**Monte Carlo simulation**:
Running the tournament thousands of times with random outcomes sampled from model probabilities, then counting how often each result occurs.
_Avoid_: random simulation, probabilistic forecast

**Tournament bracket**:
The knockout-stage tree structure where losers are eliminated. In 2026: Round of 32 → Round of 16 → QF → SF → Final.
_Avoid_: draw, knockout chart

**Group stage**:
The first phase where 48 teams play in 12 groups of 4, with top 2 + 8 best third-placed teams advancing.
_Avoid_: pool stage, first round

## Project Structure

**Pipeline**:
The end-to-end sequence: data loading → feature engineering → model training → prediction → simulation → dashboard.
_Avoid_: workflow (acceptable informally)

**Backtest**:
Running the model on a past tournament (2022 World Cup) where results are known, to validate performance before applying to 2026.
_Avoid_: retrospective test, historical validation
