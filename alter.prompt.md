# GitHub Copilot Prompt — Real-Time ML Prediction API (No .h5)

You are a **senior ML + backend engineer**.

Build a **production-ready machine learning system** that can:

- Train a mental health risk prediction model
- Persist the trained model and preprocessing artifacts
- Serve **real-time predictions via an HTTP API**

This system is based on **tabular data** and **scikit-learn models**.  
⚠️ Do **NOT** use TensorFlow, Keras, or `.h5` files.

---

## Dataset

- Source: Kaggle — _Modified depression_data with a new target column_
- CSV file: `depression_data.csv`
- Target column: `Depression Indicator` (binary: 0/1)

---

## Modeling Requirements

### 1. Preprocessing

- Load data using `pandas`
- Handle missing values safely
- Apply:
  - `OrdinalEncoder` for ordered categorical features (e.g. education level, sleep pattern, physical activity)
  - `pd.get_dummies(drop_first=True)` for nominal categorical features
  - `StandardScaler` for numeric features (e.g. age, income, number_of_children)

⚠️ Preprocessing objects **must be reusable for inference**.

---

### 2. Train-Test Split

- 80/20 split
- Use `stratify=y`
- `random_state=42`

---

### 3. Models

- Baseline: `LogisticRegression`
  - `solver="liblinear"`
  - `class_weight="balanced"`
- Primary model: `RandomForestClassifier`
  - `class_weight="balanced"`
  - Start with `n_estimators=200`

---

### 4. Hyperparameter Tuning

- Use `RandomizedSearchCV`
- 5-fold cross-validation
- `scoring="f1"`
- Tune:
  - `n_estimators`
  - `max_depth`
  - `min_samples_split`
  - `min_samples_leaf`
  - `max_features`

---

### 5. Ensemble

- Build a **Soft Voting Classifier** using:
  - Logistic Regression
  - Tuned Random Forest
  - Gradient Boosting Classifier

---

### 6. Evaluation

- Print:
  - Classification report
  - Confusion matrix
- Plot:
  - ROC curve with AUC
- Primary metric: **F1-score**
- Prioritize **Recall** (clinical relevance)

---

## Model Persistence (CRITICAL)

Persist the following using **`joblib`**:

- Trained model (`model.joblib`)
- Scaler (`scaler.joblib`)
- Encoders (`ordinal_encoder.joblib`, dummy column metadata)

❌ Do NOT use `.h5`  
❌ Do NOT retrain or refit encoders during inference

---

## API Requirements (FastAPI)

### Endpoint

- `POST /predict`

### Input

Accept JSON payload representing a single patient:

- age
- income
- education_level
- sleep_pattern
- physical_activity
- marital_status
- smoking_status
- number_of_children

### Behavior

- Load persisted model and preprocessors at startup
- Apply **the exact same preprocessing pipeline**
- Return:

```json
{
  "depression_risk": 0 or 1,
  "risk_probability": float
}
```
