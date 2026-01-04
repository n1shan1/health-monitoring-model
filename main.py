import warnings
warnings.filterwarnings('ignore')

from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.train import train_baseline, train_random_forest, tune_random_forest, train_gradient_boosting
from src.ensemble import create_voting_classifier
from src.evaluate import evaluate_model, plot_roc_curves, compare_models
from sklearn.model_selection import train_test_split

# Load and preprocess data
df = load_data('data/depression.csv')
df, ordinal_encoder, scaler, feature_cols = preprocess_data(df)

# Prepare X and y
X = df.drop('Depression Indicator', axis=1)
y = df['Depression Indicator']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# Train models
lr = train_baseline(X_train, y_train)
rf_initial = train_random_forest(X_train, y_train)
best_rf = tune_random_forest(X_train, y_train)
gb = train_gradient_boosting(X_train, y_train)

# Create ensemble
voting_clf = create_voting_classifier(lr, best_rf, gb)
voting_clf.fit(X_train, y_train)

# Predictions
lr_pred = lr.predict(X_test)
lr_proba = lr.predict_proba(X_test)[:, 1]

rf_tuned_pred = best_rf.predict(X_test)
rf_tuned_proba = best_rf.predict_proba(X_test)[:, 1]

voting_pred = voting_clf.predict(X_test)
voting_proba = voting_clf.predict_proba(X_test)[:, 1]

# Evaluate
evaluate_model(y_test, lr_pred, "Baseline: Logistic Regression")
evaluate_model(y_test, rf_tuned_pred, "Tuned Random Forest")
evaluate_model(y_test, voting_pred, "Ensemble: Soft Voting Classifier")

# Plot ROC
probas_dict = {
    'Logistic Regression': lr_proba,
    'Random Forest': rf_tuned_proba,
    'Voting Classifier': voting_proba
}
plot_roc_curves(y_test, probas_dict)

# Compare
predictions_dict = {
    'Logistic Regression': lr_pred,
    'Random Forest': rf_tuned_pred,
    'Voting Classifier': voting_pred
}
compare_models(y_test, predictions_dict, probas_dict)

# Persist artifacts
import joblib
joblib.dump(voting_clf, 'models/model.joblib')
joblib.dump(scaler, 'models/scaler.joblib')
joblib.dump(ordinal_encoder, 'models/ordinal_encoder.joblib')
joblib.dump(feature_cols, 'models/feature_cols.joblib')

print("Model and preprocessing artifacts saved to models/")