# World Cup Predictor Resources

## Knowledge

### Data & Domain
- [Dataset: International football results from 1872 to 2026 — martj42 (Kaggle)](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)
  49,000+ international match results with scores, venues, tournament type. **Primary training data.** Use for: building the match history dataframe, computing ELO ratings, and all rolling statistics.

- [Dataset: FIFA World Cup 1930–2026: Ultimate ML Dataset (Kaggle)](https://www.kaggle.com/datasets/)
  Relational dataset with 12 files — group structures, head-to-head records, pedigree indices. Use for: World Cup-specific features, 2026 group draws, and tournament structure.

- [Dataset: FIFA Rankings (Kaggle)](https://www.kaggle.com/)
  Monthly FIFA/Coca-Cola ranking points by country. Use for: FIFA ranking features and ranking difference between teams.

- [FIFA 2026 World Cup Schedule & Groups — FIFA.com](https://www.fifa.com/fifaplus/en/tournaments/mens/worldcup/canadamexicousa2026)
  Official schedule, 48 teams in 12 groups, venue assignments. Use for: building the 2026 match schedule and simulation structure.

### ML/DL Fundamentals
- [Course: Machine Learning Specialization — Andrew Ng (Coursera)](https://www.coursera.org/specializations/machine-learning-introduction)
  Gold-standard ML intro. Use for: understanding logistic regression, gradient descent, regularization, evaluation metrics. Watch Weeks 1-3 if unfamiliar.

- [Course: Practical Deep Learning for Coders — fast.ai](https://course.fast.ai/)
  Top-down, code-first DL course. Use for: understanding neural network training loops, PyTorch basics, and avoiding common pitfalls.

- [Tutorial: XGBoost Documentation — Tutorials](https://xgboost.readthedocs.io/en/latest/tutorials/index.html)
  Official XGBoost tutorials. Use for: understanding boosting, hyperparameter tuning, feature importance.

- [Library: scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
  Comprehensive docs for all classical ML. Use for: preprocessing pipelines, cross-validation, metrics, logistic regression, and Poisson regression.

- [Book: _Hands-On Machine Learning_ — Aurélien Géron (O'Reilly)](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/)
  End-to-end ML/DL with scikit-learn, Keras, and TensorFlow. Use for: deep conceptual understanding of the full pipeline. Chapters 1-7 (ML) and 10-11 (Neural Nets) are most relevant.

### Football Analytics & Feature Engineering
- [Paper: "Using ELO ratings for match result prediction" — Hvattum & Arntzen](https://ideas.repec.org/p/nms/nmsrif/3110.html)
  Foundational paper on ELO in football. Use for: understanding how to implement and tune ELO ratings for international matches.

- [Article: "Football prediction model based on ELO ratings and scoring indicators" — Sciety](https://sciety.org/)
  "Friends of ELO" approach — ELO + goals is surprisingly powerful. Use for: understanding that feature engineering matters more than model complexity.

- [Paper: "Predicting Football Match Outcomes with eXplainable ML and the Kelly Index" — arXiv](https://arxiv.org/)
  Combines ELO-based features with XGBoost and SHAP explainability. Use for: inspiration on feature engineering and model interpretability.

- [Article: "A Comprehensive Guide to Building a Soccer Prediction Model" — Towards Data Science](https://towardsdatascience.com/)
  Practical walkthrough of building a football prediction system. Use for: end-to-end pipeline structure and common pitfalls.

### Tools & Libraries
- [PyTorch Tutorials — Official](https://pytorch.org/tutorials/)
  Step-by-step PyTorch tutorials. Use for: building the neural network component (custom datasets, training loops, saving models).

- [Streamlit Documentation](https://docs.streamlit.io/)
  Official docs for building data apps. Use for: building the prediction dashboard.

- [SHAP Documentation](https://shap.readthedocs.io/)
  Model explainability library. Use for: feature importance visualizations and making the model interpretable.

- [Optuna Documentation](https://optuna.readthedocs.io/)
  Hyperparameter optimization framework. Use for: tuning XGBoost/LightGBM and neural network hyperparameters.

### Evaluation & Calibration
- [Article: "Brier Score — Measuring Probabilistic Prediction Quality"](https://en.wikipedia.org/wiki/Brier_score)
  Use for: understanding how to evaluate probability calibration (critical for Monte Carlo simulation).

- [scikit-learn: Probability Calibration](https://scikit-learn.org/stable/modules/calibration.html)
  Use for: calibrating model probabilities so Monte Carlo simulation produces reliable tournament predictions.

## Wisdom (Communities)

- [r/MachineLearning](https://reddit.com/r/MachineLearning)
  High-signal ML subreddit. Use for: staying current on techniques, getting feedback on model choices.

- [r/datascience](https://reddit.com/r/datascience)
  Practical data science community. Use for: portfolio project feedback, career advice, common mistakes to avoid.

- [Kaggle Competitions & Discussions](https://www.kaggle.com/competitions)
  Active ML competition community. Use for: seeing how others approach sports prediction problems, learning from winning solutions.

- [r/socceranalytics](https://reddit.com/r/socceranalytics)
  Niche football analytics community. Use for: domain-specific insights about what features actually matter in football prediction.

## Gaps

- No free, comprehensive dataset of **current 2026 squad rosters with player ratings**. If this surfaces, it would unlock player-level features.
- No reliable free source for **pre-match betting odds** to use as a calibration baseline. May need to manually collect from public sources during the tournament.
