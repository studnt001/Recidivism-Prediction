# Recidivism Prediction in the Criminal Justice System

---

## Project Overview

This project develops and evaluates machine learning models that predict whether a previously incarcerated individual will reoffend (recidivate). According to the Bureau of Justice Statistics, around two-thirds of state prisoners are rearrested within three years of release, costing tens of billions of dollars annually across the United States. Existing tools such as the COMPAS instrument—widely used in courtrooms—have been shown to carry meaningful racial bias, with Black defendants disproportionately classified as high risk despite similar underlying records.

This project benchmarks ten classifiers against COMPAS on the same Broward County, Florida dataset (2013–2014) used in the original ProPublica investigation. The goal is to demonstrate that open, auditable machine learning models can exceed COMPAS's predictive accuracy while making fairness metrics a first-class evaluation criterion rather than an afterthought.

**Best result:** Extra Trees — ROC-AUC **0.8901**, Accuracy **80.5%** (vs. COMPAS baseline of ~0.70 AUC / ~65% accuracy).

### Repository Structure

```
Recidivism-Prediction/
├── Code/          # Jupyter notebooks for data prep, EDA, modeling, and evaluation
└── Docs/          # Supporting documents and the final practicum report
```

---

## Setup Instructions

### Prerequisites

- Python 3.9 or higher
- Jupyter Notebook or JupyterLab

### 1. Clone the Repository

```bash
git clone https://github.com/studnt001/Recidivism-Prediction.git
cd Recidivism-Prediction
```

---

## Methodology

### Data

The raw dataset contains 18,316 records and 52 columns spanning defendant demographics, criminal history, and COMPAS scoring outputs. Records with an incomplete follow-up period (`is_recid = -1`) were removed, yielding a final analytical sample of **17,496 records** with a near-balanced class split (51.9% non-recidivist / 48.1% recidivist).

### Data Preparation

- **Datetime conversion:** Fourteen string columns were converted to `datetime64`, enabling derived temporal features such as `days_in_jail` (jail-out minus jail-in).
- **Leakage removal:** All COMPAS score columns were excluded because they are the output of the benchmarked system; including them would produce a circular and invalid model. Direct identifiers (`name`) and redundant age proxies (`dob`) were also dropped.
- **Missing values:** Numeric columns were imputed with the column median. Categorical columns were filled with the literal string `'Missing'` to preserve absence of information as a distinct category. Columns with more than 60% missing values were dropped entirely.
- **Encoding:** All remaining categorical variables were label-encoded for compatibility with scikit-learn estimators.
- After cleaning, the feature set was narrowed to **15 variables**.

### Feature Engineering

Eight new features were derived from the base 15, expanding the feature space to **23 variables**:

| Category | New Features | Rationale |
|---|---|---|
| Age | `age_bucket` (0–21, 21–25, 25–30, 30–40, 40+); `is_young_adult` (<25) | Reflects the empirically established age–crime curve |
| Juvenile history | `total_juv_charges`; `has_juv_history` (binary) | Aggregates early criminal history into a single indicator |
| Prior offenses | `priors_log1p` (log transform); `is_repeat_offender` (≥3 priors); `lifetime_charges` | Log transform compresses the right-skewed distribution of prior counts |
| Interaction | `priors_x_young` (`priors_count × is_young_adult`) | Captures elevated re-offense risk among young adults with prior records |

All features were constructed without using COMPAS scores or future recidivism outcomes to preserve methodological integrity.

### Modeling

An **80/20 stratified split** (`random_state=42`) produced 13,996 training and 3,500 test records, with stratification preserving the 52/48 class ratio. Ten classifiers were trained inside a `StandardScaler` pipeline:

Extra Trees · LightGBM · XGBoost · Random Forest · Gradient Boosting · MLP Neural Network · Decision Tree · Logistic Regression · K-Nearest Neighbors · Naive Bayes


