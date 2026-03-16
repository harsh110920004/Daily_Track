"""
Visualization Module
Creates all plots and saves results
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve
import os

def create_visualizations(df, results, models, X_test, y_test, feature_names):
    """Create and save all visualizations"""
    
    # Create output directories
    os.makedirs('outputs/figures', exist_ok=True)
    os.makedirs('outputs/results', exist_ok=True)
    
    # 1. EDA Visualizations
    create_eda_plots(df)
    
    # 2. Model comparison
    create_model_comparison(results)
    
    # 3. ROC curves
    create_roc_curves(results, y_test)
    
    # 4. Confusion matrices
    create_confusion_matrices(results)
    
    # 5. Save results to CSV
    save_results_to_csv(results)
    
    print("\nAll visualizations saved to 'outputs/figures/'")

def create_eda_plots(df):
    """Create exploratory data analysis plots"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Churn distribution
    df_temp = df.copy()
    if df_temp['Churn'].dtype == 'int64':
        churn_map = {0: 'No', 1: 'Yes'}
        df_temp['Churn'] = df_temp['Churn'].map(churn_map)
    
    df_temp['Churn'].value_counts().plot(kind='bar', ax=axes[0, 0], color=['skyblue', 'salmon'])
    axes[0, 0].set_title('Churn Distribution')
    axes[0, 0].set_xlabel('Churn')
    axes[0, 0].set_ylabel('Count')
    
    # Tenure histogram
    axes[0, 1].hist(df['tenure'], bins=30, color='lightgreen', edgecolor='black')
    axes[0, 1].set_title('Tenure Distribution')
    axes[0, 1].set_xlabel('Tenure (months)')
    
    # Monthly Charges
    axes[1, 0].hist(df['MonthlyCharges'], bins=30, color='orange', edgecolor='black')
    axes[1, 0].set_title('Monthly Charges Distribution')
    axes[1, 0].set_xlabel('Monthly Charges ($)')
    
    # Box plot for outliers
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df[numeric_cols].boxplot(ax=axes[1, 1])
    axes[1, 1].set_title('Outlier Detection')
    
    plt.tight_layout()
    plt.savefig('outputs/figures/01_eda_analysis.png', dpi=300, bbox_inches='tight')
    print("  ✓ EDA plots saved")

def create_model_comparison(results):
    """Create model performance comparison chart"""
    
    metrics_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Accuracy': [results[m]['accuracy'] for m in results],
        'Precision': [results[m]['precision'] for m in results],
        'Recall': [results[m]['recall'] for m in results],
        'F1-Score': [results[m]['f1_score'] for m in results]
    })
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(metrics_df))
    width = 0.2
    
    ax.bar(x - 1.5*width, metrics_df['Accuracy'], width, label='Accuracy')
    ax.bar(x - 0.5*width, metrics_df['Precision'], width, label='Precision')
    ax.bar(x + 0.5*width, metrics_df['Recall'], width, label='Recall')
    ax.bar(x + 1.5*width, metrics_df['F1-Score'], width, label='F1-Score')
    
    ax.set_xlabel('Models')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_df['Model'], rotation=15, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/figures/02_model_comparison.png', dpi=300, bbox_inches='tight')
    print("  ✓ Model comparison saved")

def create_roc_curves(results, y_test):
    """Create ROC curves for all models"""
    
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    
    for name, result in results.items():
        fpr, tpr, _ = roc_curve(y_test, result['probabilities'])
        plt.plot(fpr, tpr, label=f"{name} (AUC={result['auc_roc']:.3f})")
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Model Comparison')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.savefig('outputs/figures/03_roc_curves.png', dpi=300, bbox_inches='tight')
    print("  ✓ ROC curves saved")

def create_confusion_matrices(results):
    """Create confusion matrices for all models"""
    
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
    
    if n_models == 1:
        axes = [axes]
    
    for idx, (name, result) in enumerate(results.items()):
        sns.heatmap(result['confusion_matrix'], annot=True, fmt='d', 
                    cmap='Blues', ax=axes[idx], cbar=False)
        axes[idx].set_title(f'{name}')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('outputs/figures/04_confusion_matrices.png', dpi=300, bbox_inches='tight')
    print("  ✓ Confusion matrices saved")

def save_results_to_csv(results):
    """Save evaluation results to CSV"""
    
    results_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Accuracy': [results[m]['accuracy'] for m in results],
        'Precision': [results[m]['precision'] for m in results],
        'Recall': [results[m]['recall'] for m in results],
        'F1-Score': [results[m]['f1_score'] for m in results],
        'AUC-ROC': [results[m]['auc_roc'] for m in results]
    })
    
    results_df.to_csv('outputs/results/model_evaluation.csv', index=False)
    print("  ✓ Results saved to CSV")
