import pandas as pd

def load_data(filepath):
    """Load the depression dataset from CSV."""
    df = pd.read_csv(filepath)
    print("Dataset shape:", df.shape)
    print("Target distribution:")
    print(df['Depression Indicator'].value_counts(normalize=True))
    return df