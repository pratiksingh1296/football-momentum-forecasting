# Football Momentum Forecasting

A DistilBERT based NLP system that classifies football match momentum from live commentary text, achieving 91% macro F1 (up from a 72% baseline) through systematic, hypothesis driven experimentation.

**Live demo:** [https://football-momentum-forecasting.streamlit.app/]
**Model checkpoint:** [huggingface.co/leoken/football-momentum-distilbert](https://huggingface.co/leoken/football-momentum-distilbert)

---

## Overview

This project predicts whether a football match is currently trending toward the home team, the away team, or neither (a balanced state), using only 5 minute windows of real match commentary text. The task is framed as 3 class classification: **Home Dominant**, **Away Dominant**, or **Balanced**.

The dataset is built from a public Kaggle dataset of 941,009 events across 9,074 matches from the top 5 European leagues (2011 to 2017). Labels are derived from a weighted, event based momentum score computed independently for each 5 minute window and are not present in the raw data.

Rather than jumping straight to a final model, this project is structured as a series of controlled experiments, each testing one specific hypothesis about what the model needs in order to read momentum correctly. The result is not just a working classifier, but a documented account of what worked, what did not, and why.

---

## Results Summary

| Experiment | Description | Macro F1 | Outcome |
|---|---|---|---|
| Baseline | LSTM on raw commentary text | 0.54 | Establishes a starting point |
| Baseline (DistilBERT) | DistilBERT on raw commentary text, no side tagging | 0.72 | Strong text signal, but confuses Home vs Away |
| **Experiment A** | DistilBERT with explicit Home/Away side tagging | **0.9077** | Resolves the Home/Away confound; primary deployed model |
| Experiment B | A + player/team name anonymization | 0.9103 | No meaningful improvement over A (within noise) |
| Experiment C | Engineered numeric match state features alone (LightGBM / Logistic Regression) | 0.39 | Numeric features alone are insufficient |
| Experiment D | A + numeric features fused into DistilBERT | 0.9176 | Small, consistent improvement across all classes |
| Experiment E | Window size sensitivity (3 min / 5 min / 10 min) | 0.98 (3 min, inflated) | Headline number misleading; see below |

**Deployed model: Experiment A.** Despite Experiment D's modest edge, A remains the safer, most thoroughly validated choice for production use. See "Why Experiment A, not D or E" below.

---

## The Core Problem and Diagnosis

Commentary text describes events using team and player names, but never states which side is the home team and which is the away team. A baseline DistilBERT model trained on raw commentary text reached only 72% macro F1, and error analysis showed most of its mistakes were Home/Away direction confusion, not genuine misreading of momentum.

The likely cause: without knowing which named team is "home," the model can only guess Home vs Away direction by memorizing which teams tend to appear as home or away in the training data, a spurious shortcut that does not generalize.

**Fix (Experiment A):** each event in the input text is explicitly prefixed with `[HOME_TEAM]` or `[AWAY_TEAM]` before being fed to the model, for example:

```
[HOME_TEAM] Goal! Borussia Dortmund 1, Hamburg 0. Kevin Grosskreutz (Borussia Dortmund) left footed shot from the left side of the box to the bottom right corner.
```

This single change took macro F1 from 0.72 to 0.9077, and reduced Home/Away direction confusion errors by approximately 96%.

---

## Experiments in Detail

### Experiment A: Home/Away Side Tagging
Tags every event with its side before tokenization. This isolates a single variable (does the model need explicit side information) and produces the largest single improvement in the whole project. Deployed model.

### Experiment B: Entity Anonymization
Tests whether the model was relying on player and team name identity as a shortcut, on top of A's tags. Player and team mentions were replaced with `HOME_PLAYER` / `AWAY_PLAYER` / `HOME_TEAM` / `AWAY_TEAM` placeholders directly in the text, using a custom letter by letter matching scheme to handle inconsistent name formatting in the source data (encoding issues, merged tokens, abbreviated club names).

Result: macro F1 changed by +0.0026, well within expected run to run training variance for a single training run. Concluded as a genuine null result: A's side tags already provide sufficient signal, and residual name identity is not doing meaningful work.

### Experiment C: Structured Numeric Features
Tests whether engineered numeric match state features (cumulative shots, shots on target, cards, substitutions, time since last event, lagged momentum, etc, all computed only from information available before the current window, to avoid label leakage) can predict momentum on their own, using Logistic Regression and LightGBM.

Result: macro F1 of approximately 0.39, barely above the naive majority class baseline. LightGBM and Logistic Regression performed almost identically, indicating the ceiling is a signal limitation, not a model capacity limitation. Feature importance showed simple cumulative counts and lagged momentum dominated over more elaborate engineered features (shot quality, set pieces, recency), suggesting aggregation itself, not feature choice, is the bottleneck: numeric aggregates discard the sequential and contextual detail that text preserves.

### Experiment D: Text and Numeric Fusion
Combines Experiment A's text representation with Experiment C's numeric features in a single architecture: DistilBERT's pooled output is concatenated with a small linear projection of the numeric features before the final classifier.

Result: macro F1 of 0.9176, a consistent roughly 1 percentage point improvement across every class and every aggregate metric, unlike Experiment B's mixed, noise scale movement. This indicates the numeric features contribute complementary signal on top of text, even though they were insufficient alone (Experiment C). Not selected for deployment due to added pipeline complexity (a fitted scaler, 31 live engineered features, two data pipelines to keep in sync) relative to the gain, and because the result is based on a single training run with no repeated seed confirmation.

### Experiment E: Window Size Sensitivity
Tests whether a shorter (3 minute) or longer (10 minute) window changes task difficulty, holding all other variables fixed.

The 3 minute window produced a dramatic aggregate macro F1 of 0.98, which further analysis showed was substantially inflated: 3 minute windows are dominated by short, sparse, often single sided windows (24% of test windows mention only one side, and these are classified with 99.98% accuracy, since the side tag alone is nearly sufficient). Restricting evaluation to genuinely comparable, multi event windows narrows the gap to 97.9% vs Experiment A's 91%, a real but far smaller and less certain effect than the raw aggregate suggests.

The 10 minute window produced a macro F1 of 0.65, confounded by both a harder classification task and roughly half the training data compared to A.

**Conclusion:** On a fair comparison restricted to windows where both sides have activity, the 3 minute configuration is genuinely easier to classify correctly than the 5 minute configuration (97.9% vs 91% accuracy), suggesting shorter windows do carry a real learnability advantage, consistent with less time for momentum to reverse or ambiguity to accumulate within a single window. However, this finding does not by itself justify moving away from the 5 minute configuration for deployment: the 3 minute dataset is also dominated by sparse, near trivial single sided windows whose real world frequency in unseen matches is uncertain, and the result reflects a single training run without repeated seed confirmation. Window size sensitivity is a genuine and promising direction for further work, evaluated here honestly rather than adopted outright.

### Boundary Sensitivity Analysis
A follow up post hoc analysis on Experiment A's own test predictions found that 83.7% of near label boundary errors (within 0.5 momentum score of the +/-2 Home/Away/Balanced threshold) involve the Balanced class, and that error rate declines monotonically the further a window's true momentum score sits from the boundary. This indicates a meaningful share of the model's residual errors, particularly on the Balanced class, are attributable to inherent ambiguity in the label threshold itself, not solely to model weakness.

---

## Why Experiment A, not D or E

- **D** offers a small (about 1 point) improvement at the cost of a second data pipeline, a fitted scaler that must travel with the model, and a result validated only once. The added complexity is not proportionate to the gain for a deployed demo.
- **E's** headline number does not hold up under scrutiny: once composition effects (sparse, near trivial single sided windows) are accounted for, the genuine improvement is much smaller, and it is unclear how this generalizes to unseen matches with unknown windowing distributions.
- **A** is the most thoroughly validated result across every dimension checked in this project (confusion matrix analysis, boundary sensitivity analysis, and consistent behavior across every downstream experiment), and remains the safest choice for a public facing demo.

---

## Repository Structure

```
football-momentum-forecasting/
├── data/
│   ├── raw/                     # original Kaggle event data
│   └── processed/
│       ├── baseline/
│       ├── exp_a_prefix/
│       ├── exp_b_entities/
│       ├── exp_c_tabular/
│       ├── exp_d_hybrid/
│       └── exp_e_window_sens/
├── models/                      # saved model checkpoints and scalers per experiment
├── notebooks/
│   ├── 00_data_sanity.ipynb
│   ├── 01_momentum_labels.ipynb
│   ├── 02_results_analysis.ipynb
│   ├── 03_exp_a_event_side.ipynb
│   ├── 04_exp_b_entity_normalization.ipynb
│   ├── 05_exp_c_tabular.ipynb
│   ├── 06_exp_d_hybrid.ipynb
│   └── 07_exp_e_window_sens.ipynb
├── reports/
│   ├── figures/
│   └── metrics/
├── src/
│   ├── config.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
├── app/
│   ├── Match_Replay.py          # Streamlit entrypoint
│   └── pages/
│       └── Try_The_Model.py
└── README.md
```

---

## Methodology Notes

- **Labels:** for each 5 minute window, a momentum score is computed as the weighted sum of home side events minus away side events (Attempt: +3, Corner: +1, Goal: +5, Yellow Card: -1, Red Card: -3, Offside: -0.5). Momentum >= 2 is Home Dominant, <= -2 is Away Dominant, otherwise Balanced.
- **Splitting:** all experiments share an identical match level (not window level) 70/15/15 train/val/test split, verified by exact match ID set comparison across every experiment, to guarantee results are comparable and no match's windows leak across splits.
- **Leakage discipline:** numeric features in Experiments C and D are computed only from information available strictly before the window being labeled (lagged momentum, cumulative counts up to but not including the current window), to avoid trivially reconstructing the label from its own inputs.
- **Honesty about uncertainty:** every experiment in this project was trained once. Where an observed difference could plausibly be explained by ordinary training run variance rather than a genuine effect (Experiment B), this is stated explicitly rather than overclaimed.

---

## Tech Stack

- **Model:** DistilBERT (Hugging Face Transformers), PyTorch
- **Tabular baselines:** scikit learn, LightGBM
- **Data processing:** pandas, NumPy
- **Deployment:** Streamlit, Hugging Face Hub (model hosting)
- **Visualization:** Plotly, Matplotlib, Seaborn

---

## Future Work

- Repeat Experiment D across multiple random seeds to confirm the improvement is not a favorable initialization.
- Run a controlled window size comparison restricted to multi event windows only, to isolate the genuine effect of window size from task composition effects (Experiment E).
- Explore soft or ordinal label formulations (continuous momentum score rather than a hard 3 way threshold) to address the boundary ambiguity identified in the boundary sensitivity analysis.
- Investigate finer grained (sub 3 minute) windows using richer input sources, such as tracking or possession data, where signal density does not collapse the way it does with sparse commentary text.

---

## Author

Pratik Singh
[LinkedIn](https://linkedin.com/in/pratik-singh-ds) | [GitHub](https://github.com/pratiksingh1296)