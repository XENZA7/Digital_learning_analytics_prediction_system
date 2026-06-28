# Digital Learning Analytics & Prediction System

A full end-to-end machine learning and data science project that analyses online learner behaviour and predicts **course completion** and **mastery scores** across MOOC platforms and digital learning apps. The project covers the complete data science lifecycle — from exploratory analysis and statistical profiling through feature engineering, predictive modelling, and model explainability — built with production-ready ML engineering practices.

---

## Results

### Classification — Course Completion Prediction

| Model | Accuracy | F1-Score | ROC-AUC |
|---|---|---|---|
| Logistic Regression (baseline) | 95.4% | 0.929 | 0.993 |
| Random Forest (tuned) | 92.8% | 0.874 | **0.974** |

> **Note on data leakage:** Initial experiments included post-outcome features (`skill_post_score`, `time_to_mastery_hours`, `learning_efficiency_score`, etc.) that would not be available at inference time. These were identified and documented as a known limitation. In a production deployment, the model would be retrained using only features available at enrollment time, which is expected to bring ROC-AUC to the 0.80–0.85 range — still competitive for student outcome prediction.

### Regression — Mastery Score Prediction

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Ridge Regression (baseline) | 4.34 | 5.73 | 0.767 |
| Random Forest (tuned) | **0.83** | **2.29** | **0.963** |

---

## Project Structure

```
Digital_learning_analytics_prediction_system/
│
├── digital_learning.ipynb       # Main notebook — full pipeline
├── requirements.txt             # Python dependencies
│
├── Data/
│   └── digital_learning_analytics.csv   # 100,000 learner records, 43 features
│
└── artifacts/                   # Saved preprocessing objects and models
    ├── imputer.joblib
    ├── label_encoders.joblib
    ├── scaler_regression.joblib
    ├── scaler_classification.joblib
    ├── rf_classifier.joblib
    ├── rf_regressor.joblib
    └── ridge_regressor.joblib
```

---

## Dataset

- **100,000** learner records spanning **2020–2025**
- **43 features** across six domains:
  - Learner demographics (age, education level, country, employment status)
  - Digital app behaviour (daily minutes, session frequency, quiz scores, gamification)
  - Essay performance (word count, grammar errors, vocabulary richness, coherence)
  - MOOC engagement (video completion %, assignment submission rate, forum posts)
  - Adaptive learning signals (knowledge gaps, remediation modules, content difficulty)
  - Temporal features (enrollment date, activity duration, learning velocity)
- **Class imbalance:** 30.6% course completion vs 69.4% non-completion
- **Dual targets:** `course_completed` (binary classification) and `mastery_score` (continuous regression, range 1.7–100, mean 47.8, skewness 0.215)

---

## Data Science Analysis

### Exploratory Data Analysis
A thorough EDA was conducted across all 43 features before any modelling:

- **Univariate analysis** — distribution plots and summary statistics for all numerical features; frequency counts for all categorical features
- **Outlier detection** — IQR-based box plots across 11 key numerical features including `daily_app_minutes`, `total_learning_hours`, `digital_literacy_score`, and `engagement_consistency`
- **Target distribution analysis** — class imbalance quantified (30.6% vs 69.4%); mastery score distribution assessed for skewness (0.215 — approximately normal)
- **Correlation analysis** — heatmap of 14 key numerical features to identify multicollinearity and feature relationships with both targets
- **Bivariate analysis** — course completion rates broken down by education level, employment status, and digital literacy score; skill improvement distributions compared across completion groups

### Temporal Analysis
- **Monthly enrollment trends** tracked from 2020 to 2025, revealing learner volume patterns over time
- **Activity duration distribution** — days between enrollment and last activity analysed across the full cohort
- **Cross-domain relationship analysis** — MOOC platform vs course category, and education level vs app category visualised to understand multi-domain data patterns

### Statistical Profiling
- Missing value analysis: `gamification_engagement` (2,490 missing), `essay_vocabulary_richness` (1,973 missing), `essay_coherence_score` (2,022 missing) — all handled via median imputation fitted on training data only
- Zero duplicate records confirmed across all 100,000 entries
- Residual analysis on regression predictions using LOWESS smoothing to validate model assumptions

---

## ML Engineering Highlights

### Time-Aware Data Splitting
Data is split by enrollment year rather than randomly — a critical practice for temporal datasets that prevents future information from leaking into training.

```
Train:      2020 – 2023    (66,632 records)
Validation: 2024           (16,743 records)
Test:       2025           (16,625 records)
```

### Leakage-Free Preprocessing Pipeline
All preprocessing steps — imputation, outlier capping, categorical encoding, and scaling — are **fitted exclusively on training data** and applied to validation and test sets. This prevents data leakage and simulates real deployment conditions.

### Feature Engineering
Five domain-meaningful features derived from raw data:

| Feature | Formula | Insight |
|---|---|---|
| `skill_improvement` | `skill_post_score − skill_pre_score` | Measures actual learning gain |
| `engagement_score` | Mean of video completion, assignment rate, consistency | Composite engagement signal |
| `learning_velocity` | `total_learning_hours / course_duration_weeks` | Pace of learning relative to course length |
| `essay_error_rate` | `grammar_errors / word_count` | Normalised writing quality signal |
| `activity_duration_days` | `last_activity_date − enrollment_date` | Learner persistence measure |

### Unseen Category Handling
Label encoders are fitted with an `'unknown'` class to gracefully handle new categorical values at inference time — a common production edge case.

### Class Imbalance Handling
`class_weight='balanced'` applied to both Logistic Regression and Random Forest classifiers, automatically adjusting sample weights inversely proportional to class frequency (30.6% positive class).

### Model Explainability (SHAP)
SHAP TreeExplainer applied to both the classification and regression models to identify the most influential features driving predictions — making the system interpretable for educational institutions that need to understand and act on predictions.

### Artifact Persistence
All preprocessing objects and trained models are saved to the `artifacts/` directory using `joblib`, enabling inference without retraining.

---

## Setup & Usage

**1. Clone the repository**
```bash
git clone https://github.com/XENZA7/Digital_learning_analytics_prediction_system.git
cd Digital_learning_analytics_prediction_system
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the notebook**

Open `digital_learning.ipynb` in Jupyter or VS Code and run all cells. The notebook is structured in sequential sections:

1. Data Inspection
2. Target Distribution Analysis
3. Exploratory Data Analysis (EDA)
4. Advanced EDA — Temporal & Cross-Domain Analysis
5. Preprocessing & Feature Engineering
6. Model Building — Classification
7. Model Building — Regression
8. Model Evaluation & Comparison

---

## Tech Stack

| Category | Tools |
|---|---|
| Data manipulation | pandas, numpy |
| Visualisation | matplotlib, seaborn |
| Statistical analysis | statsmodels, scipy (via sklearn) |
| Modelling | scikit-learn (LogisticRegression, Ridge, RandomForestClassifier, RandomForestRegressor, RandomizedSearchCV) |
| Explainability | SHAP (TreeExplainer) |
| Artifact persistence | joblib |

---

## Known Limitations

- The dataset is synthetic/benchmark in nature. Results may differ on real institutional data where feature distributions and completion rates vary significantly.
- Hyperparameter tuning used a small subsample (1,000 records) with limited iterations (`n_iter=3`) for speed. A broader search on the full training set would likely improve the tuned model further.
- Post-outcome features are included in the current feature set. Production deployment would require restricting inputs to features available at enrollment time only.

---

## Author

**Siyamthanda Xenza** — Final Year Student | Aspiring Data Scientist & ML Engineer  
📍 South Africa  
🔗 [GitHub](https://github.com/XENZA7)
