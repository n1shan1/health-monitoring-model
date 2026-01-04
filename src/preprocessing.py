import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

def preprocess_data(df):
    """Handle missing values and apply encoding/scaling."""
    # Select relevant columns as per API requirements
    relevant_cols = [
        'Age', 'Income', 'Education Level', 'Sleep Patterns', 
        'Physical Activity Level', 'Marital Status', 'Smoking Status', 
        'Number of Children', 'Depression Indicator'
    ]
    df = df[relevant_cols]
    
    # Drop rows with missing target
    df = df.dropna(subset=['Depression Indicator'])

    # Define columns
    ordinal_features = ['Education Level', 'Sleep Patterns', 'Physical Activity Level']
    nominal_features = ['Marital Status', 'Smoking Status']
    numerical_features = ['Age', 'Income', 'Number of Children']

    # Impute missing values
    categorical_cols = ordinal_features + nominal_features
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])

    for col in numerical_features:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    # Encode target
    df['Depression Indicator'] = df['Depression Indicator'].map({'Yes': 1, 'No': 0})

    # Ordinal encoding
    ordinal_encoder = OrdinalEncoder()
    df[ordinal_features] = ordinal_encoder.fit_transform(df[ordinal_features])

    # One-hot encoding
    df = pd.get_dummies(df, columns=nominal_features, drop_first=True)

    # Standard scaling
    scaler = StandardScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    # Get feature columns after preprocessing
    feature_cols = df.drop('Depression Indicator', axis=1).columns.tolist()

    return df, ordinal_encoder, scaler, feature_cols

    # Encode target
    df['Depression Indicator'] = df['Depression Indicator'].map({'Yes': 1, 'No': 0})

    # Encode binary columns
    binary_cols = ['History of Mental Illness', 'History of Substance Abuse', 'Family History of Depression', 'Chronic Medical Conditions']
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0})

    # Ordinal encoding
    ordinal_encoder = OrdinalEncoder()
    df[ordinal_features] = ordinal_encoder.fit_transform(df[ordinal_features])

    # One-hot encoding
    df = pd.get_dummies(df, columns=nominal_features, drop_first=True)

    # Standard scaling
    scaler = StandardScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    return df