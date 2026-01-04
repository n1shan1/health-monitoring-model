import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

def preprocess_data(df):
    """Handle missing values and apply encoding/scaling."""
    # Drop unnecessary columns
    if 'Name' in df.columns:
        df.drop('Name', axis=1, inplace=True)
    
    # Drop rows with missing target
    df = df.dropna(subset=['Depression Indicator'])

    # Define columns
    ordinal_features = ['Education Level', 'Sleep Patterns', 'Physical Activity Level', 'Smoking Status', 'Employment Status', 'Alcohol Consumption', 'Dietary Habits']
    nominal_features = ['Marital Status']
    numerical_features = ['Age', 'Income', 'Number of Children']
    
    # Get all categorical columns for imputation
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if 'Depression Indicator' in categorical_cols:
        categorical_cols.remove('Depression Indicator')

    # Impute missing values
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])

    for col in numerical_features:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

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