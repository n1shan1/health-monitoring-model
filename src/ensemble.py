from sklearn.ensemble import VotingClassifier

def create_voting_classifier(lr, rf, gb):
    """Create and train Soft Voting Classifier."""
    voting_clf = VotingClassifier(
        estimators=[
            ('lr', lr),
            ('rf', rf),
            ('gb', gb)
        ],
        voting='soft'
    )
    return voting_clf