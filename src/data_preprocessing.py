"""
Data Preprocessing Module
Handles data loading, cleaning, and feature engineering
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

def load_and_clean_data(filepath):
    """Load and perform initial cleaning of the dataset"""
    
    # Check if file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    
    # Load data
    df = pd.read_csv(filepath)
    
    # Basic info
    print(f"\nOriginal dataset shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum().sum()}")
    
    # Handle TotalCharges column
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    
    # Drop customerID
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)
    
    print(f"Cleaned dataset shape: {df.shape}")
    
    return df

def preprocess_features(df):
    """Preprocess features for modeling"""
    
    # Make a copy
    df_processed = df.copy()
    
    # Encode binary variables
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 
                   'PaperlessBilling', 'Churn']
    
    label_encoder = LabelEncoder()
    for col in binary_cols:
        if col in df_processed.columns:
            df_processed[col] = label_encoder.fit_transform(df_processed[col])
    
    # One-hot encode multi-category variables
    multi_cat_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity',
                      'OnlineBackup', 'DeviceProtection', 'TechSupport',
                      'StreamingTV', 'StreamingMovies', 'Contract',
                      'PaymentMethod']
    
    # Filter only existing columns
    multi_cat_cols = [col for col in multi_cat_cols if col in df_processed.columns]
    
    df_encoded = pd.get_dummies(df_processed, columns=multi_cat_cols, drop_first=True)
    
    # Split features and target
    X = df_encoded.drop('Churn', axis=1)
    y = df_encoded['Churn']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns.tolist()
