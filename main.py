"""
Customer Churn Prediction - Main Script
CSE3141 - Predictive Analysis Project
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from src.data_preprocessing import load_and_clean_data, preprocess_features
from src.model_training import train_models, evaluate_models
from src.visualization import create_visualizations

def main():
    print("="*70)
    print("CUSTOMER CHURN PREDICTION PROJECT")
    print("="*70)
    
    # Step 1: Load and clean data
    print("\n[1/4] Loading and cleaning data...")
    df = load_and_clean_data('data/raw/Telco-Customer-Churn.csv')
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Step 2: Preprocess features
    print("\n[2/4] Preprocessing features...")
    X_train, X_test, y_train, y_test, feature_names = preprocess_features(df)
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Step 3: Train models
    print("\n[3/4] Training models...")
    models = train_models(X_train, y_train)
    print(f"Models trained: {len(models)}")
    
    # Step 4: Evaluate models
    print("\n[4/4] Evaluating models and creating visualizations...")
    results = evaluate_models(models, X_test, y_test)
    
    # Create visualizations
    create_visualizations(df, results, models, X_test, y_test, feature_names)
    
    print("\n" + "="*70)
    print("PROJECT COMPLETED SUCCESSFULLY!")
    print("Check 'outputs/figures/' for visualizations")
    print("Check 'outputs/results/' for detailed results")
    print("="*70)

if __name__ == "__main__":
    main()
