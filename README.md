# Mental Health Risk Prediction Model

This project implements a machine learning system for predicting mental health risk using the Anthony Therrien Depression Dataset.

## Project Structure

```
mental-health-ml/
│
├── data/
│   └── depression_data.csv
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── ensemble.py
│
├── models/
│   └── best_random_forest.pkl
│
├── plots/
│   ├── roc_curve.png
│   └── confusion_matrix.png
│
├── main.py
├── requirements.txt
└── README.md
```

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Place the dataset `depression_data.csv` in the `data/` directory.

3. Run the main script:
   ```bash
   python main.py
   ```

## Methodology

### Data Preprocessing
- Ordinal encoding for ordered categorical variables (Education Level, Sleep Patterns, Physical Activity Level)
- One-hot encoding for nominal variables (Marital Status, Smoking Status)
- Standardization for numerical features (Age, Income, Number of Children)
- Class imbalance handled using `class_weight='balanced'`

### Models
- **Baseline**: Logistic Regression
- **Primary**: Tuned Random Forest (optimized for F1-score)
- **Ensemble**: Soft Voting Classifier (LR + RF + Gradient Boosting)

### Evaluation
- Primary metric: F1-Score
- Secondary: Recall (clinical relevance)
- ROC-AUC for discriminative performance

## Results

The system achieves high performance with F1-score > 0.98 on the test set, prioritizing recall to minimize false negatives in clinical screening.