"""
Model Training Module
Trains and evaluates multiple classification models
"""

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import numpy as np

def train_models(X_train, y_train):
    """Train multiple classification models"""
    
    models = {}
    
    # Logistic Regression
    print("  - Training Logistic Regression...")
    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    log_reg.fit(X_train, y_train)
    models['Logistic Regression'] = log_reg
    
    # Linear Discriminant Analysis
    print("  - Training LDA...")
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)
    models['LDA'] = lda
    
    # Quadratic Discriminant Analysis
    print("  - Training QDA...")
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(X_train, y_train)
    models['QDA'] = qda
    
    # Hyperparameter tuning for Logistic Regression
    print("  - Tuning Logistic Regression...")
    param_grid = {
        'C': [0.01, 0.1, 1, 10],
        'penalty': ['l2'],
        'solver': ['lbfgs']
    }
    
    grid_search = GridSearchCV(
        LogisticRegression(max_iter=1000, random_state=42),
        param_grid, cv=5, scoring='accuracy', n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    models['Tuned Logistic Regression'] = grid_search.best_estimator_
    
    print(f"  Best parameters: {grid_search.best_params_}")
    
    return models

def evaluate_models(models, X_test, y_test):
    """Evaluate all trained models"""
    
    results = {}
    
    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'auc_roc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        print(f"  Accuracy: {results[name]['accuracy']:.4f}")
        print(f"  F1-Score: {results[name]['f1_score']:.4f}")
        print(f"  AUC-ROC: {results[name]['auc_roc']:.4f}")
    
    return results
