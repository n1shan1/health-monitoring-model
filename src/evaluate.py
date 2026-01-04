from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, f1_score, recall_score
import matplotlib.pyplot as plt
import pandas as pd

def evaluate_model(y_test, y_pred, model_name):
    """Print classification report and confusion matrix."""
    print(f"\n=== {model_name} ===")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

def plot_roc_curves(y_test, probas_dict):
    """Plot ROC curves for multiple models."""
    plt.figure(figsize=(10, 8))
    for name, proba in probas_dict.items():
        fpr, tpr, _ = roc_curve(y_test, proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig('plots/roc_curve.png', dpi=300, bbox_inches='tight')
    plt.show()

def compare_models(y_test, predictions_dict, probas_dict):
    """Create comparison table."""
    models = list(predictions_dict.keys())
    f1_scores = [f1_score(y_test, pred) for pred in predictions_dict.values()]
    recalls = [recall_score(y_test, pred) for pred in predictions_dict.values()]
    aucs = [auc(*roc_curve(y_test, proba)[:2]) for proba in probas_dict.values()]

    summary = {
        'Model': models,
        'F1-Score': f1_scores,
        'Recall': recalls,
        'AUC': aucs
    }

    summary_df = pd.DataFrame(summary)
    print("\n=== Model Comparison ===")
    print(summary_df.round(4))
    return summary_df