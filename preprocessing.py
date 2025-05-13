import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def preprocess_data(df):
    # Encode categorical 'risk_level'
    le = LabelEncoder()
    df['risk_level'] = le.fit_transform(df['risk_level'])

    # Scale numerical features
    scaler = StandardScaler()
    numerical_features = ['budget', 'team_size', 'duration']
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    return df, le, scaler