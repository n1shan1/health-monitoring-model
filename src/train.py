from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import joblib

def train_baseline(X_train, y_train):
    """Train baseline Logistic Regression."""
    lr = LogisticRegression(solver='liblinear', class_weight='balanced', random_state=42)
    lr.fit(X_train, y_train)
    return lr

def train_random_forest(X_train, y_train):
    """Train initial Random Forest."""
    rf = RandomForestClassifier(class_weight='balanced', n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    return rf

def tune_random_forest(X_train, y_train):
    """Tune Random Forest using RandomizedSearchCV."""
    param_distributions = {
        "n_estimators": [200, 400, 600, 800, 1000, 1200, 1500],
        "max_depth": [20, 25, 30, 35, 40],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"]
    }

    rf_search = RandomizedSearchCV(
        estimator=RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1),
        param_distributions=param_distributions,
        n_iter=30,
        scoring="f1",
        cv=5,
        verbose=2,
        random_state=42,
        n_jobs=-1
    )

    rf_search.fit(X_train, y_train)
    best_rf = rf_search.best_estimator_

    # Save best model
    joblib.dump(best_rf, 'models/best_random_forest.pkl')

    print("Best RF params:", rf_search.best_params_)
    print("Best CV F1:", rf_search.best_score_)

    return best_rf

def train_gradient_boosting(X_train, y_train):
    """Train Gradient Boosting Classifier."""
    gb = GradientBoostingClassifier(random_state=42)
    gb.fit(X_train, y_train)
    return gb