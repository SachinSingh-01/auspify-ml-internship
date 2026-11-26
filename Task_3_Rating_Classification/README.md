# Netflix Audience Rating Classification (Task 3)

## Overview
A multi-class machine learning classification pipeline built to predict the audience maturity rating (`TV-MA`, `TV-14`, `TV-PG`, `R`, `PG-13`, `TV-Y7`, `TV-Y`, `PG`, `TV-G`, `NR`, `G`) of Netflix titles based on content type, production country, release year, duration, and genres.

## Dataset & Preprocessing
- **Dataset Size:** 8,778 titles across 11 rating categories (filtered out ultra-rare classes with fewer than 10 samples: `TV-Y7-FV`, `NC-17`, `UR` for stratified cross-validation).
- **Train/Test Split:** 80% Training (7,022 rows) / 20% Testing (1,756 rows) using stratified sampling.
- **Feature Engineering (62 Features):**
  - Numeric extraction of `duration` (`duration_num`) alongside `release_year`.
  - One-hot encoding of `type` and top 15 countries (with infrequent countries grouped as `Other`).
  - Multi-label binarization (`MultiLabelBinarizer`) of the 42 genre tags in `listed_in`.

## Model Comparison & Hyperparameter Tuning
| Model | Test Accuracy | Macro F1 | Weighted F1 |
| :--- | :---: | :---: | :---: |
| Logistic Regression | 43.56% | 0.16 | 0.37 |
| Decision Tree | 45.67% | 0.38 | 0.45 |
| Random Forest (Default) | 51.77% | 0.43 | 0.51 |
| **Random Forest (Tuned via RandomizedSearchCV)** | **55.18%** | **0.46** | **0.52** |

**Best Hyperparameters:** `n_estimators=300`, `min_samples_split=5`, `min_samples_leaf=2`, `max_depth=None` (5-Fold CV Weighted F1: **0.5422**).

## Key Insights
- `duration_num` and `release_year` are the strongest overall predictors, separating theatrical film runtimes from episodic TV series and capturing shifts in modern streaming ratings.
- Genre indicators (`Children & Family Movies`, `Kids' TV`, `International Movies`, `Stand-Up Comedy`) strongly separate youth ratings (`TV-Y`, `TV-Y7`, `PG`) from mature categories (`TV-MA`, `R`).

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   