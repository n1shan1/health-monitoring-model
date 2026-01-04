Prompt:
You are an expert ML engineer. Write a complete Python script to build a mental health prediction model using the Modified depression_data with a new target column dataset from Kaggle.

The CSV file is named depression_data.csv. The target column is Depression Indicator (binary: 0/1).

Data Preprocessing:

Load the dataset with pandas.

Handle missing values reasonably (drop or impute).

Apply:

OrdinalEncoder to ordinal categorical features (e.g., education, sleep quality, activity level if present).

pd.get_dummies(drop_first=True) for nominal categorical features.

StandardScaler for numeric features (e.g., age, income, etc.).

Train-Test Split:

Use train_test_split with test_size=0.2, stratify=y, and random_state=42.

Models:

Train a baseline Logistic Regression (solver='liblinear', class_weight='balanced').

Train a RandomForestClassifier with class_weight='balanced' and initial n_estimators=200.

Hyperparameter Tuning:

Use RandomizedSearchCV with 5-fold cross-validation optimizing scoring='f1'.

Search space includes: n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features.

Ensemble:

Build a Soft Voting Classifier combining Logistic Regression, tuned Random Forest, and Gradient Boosting.

Evaluation:

Print classification report, confusion matrix, and ROC curve with AUC.

Focus on F1-score and recall due to class imbalance.

Use only: pandas, scikit-learn, matplotlib.

📌 Adaptation Notes

✔ The target column in this dataset is Depression Indicator (binary classification).
Kaggle

✔ You can treat this dataset exactly like the original Anthony Therrien dataset you planned — just replace the target variable in the code with Depression Indicator.

✔ If the dataset doesn’t have specific ordinal features (like sleep or activity levels), you can adjust by treating those fields as either encoded categorical features or numerical, depending on the actual column types once you inspect the CSV.

📝 Quick Example: How to Rename Target in Your Code

In your ML script, replace:

y = df["History of Mental Illness"]

with:

y = df["Depression Indicator"]

and keep the rest of the pipeline the same.
