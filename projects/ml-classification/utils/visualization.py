"""
Visualization utilities for data exploration and model evaluation.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class DataVisualizer:
    """
    Utility class for data visualization and exploration.
    """
    
    def __init__(self, style='seaborn-v0_8', palette='husl'):
        """
        Initialize DataVisualizer.
        
        Args:
            style (str): Matplotlib style
            palette (str): Seaborn color palette
        """
        plt.style.use(style)
        sns.set_palette(palette)
        
    def plot_distribution(self, data, columns=None, figsize=(15, 10)):
        """
        Plot distribution of numerical columns.
        
        Args:
            data (pd.DataFrame): Input data
            columns (list): Columns to plot (if None, plot all numerical)
            figsize (tuple): Figure size
        """
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns
            
        n_cols = 3
        n_rows = (len(columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_rows > 1 else [axes]
        
        for i, col in enumerate(columns):
            if i < len(axes):
                data[col].hist(bins=30, ax=axes[i], alpha=0.7)
                axes[i].set_title(f'Distribution of {col}')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frequency')
                
        # Hide empty subplots
        for i in range(len(columns), len(axes)):
            axes[i].set_visible(False)
            
        plt.tight_layout()
        plt.show()
        
    def plot_correlation_matrix(self, data, figsize=(12, 10)):
        """
        Plot correlation matrix heatmap.
        
        Args:
            data (pd.DataFrame): Input data
            figsize (tuple): Figure size
        """
        plt.figure(figsize=figsize)
        correlation_matrix = data.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f')
        plt.title('Correlation Matrix')
        plt.show()
        
    def plot_pairplot(self, data, target_column=None, sample_size=1000):
        """
        Plot pairplot for numerical columns.
        
        Args:
            data (pd.DataFrame): Input data
            target_column (str): Target column for coloring
            sample_size (int): Sample size for large datasets
        """
        if len(data) > sample_size:
            data_sample = data.sample(sample_size)
        else:
            data_sample = data
            
        if target_column:
            sns.pairplot(data_sample, hue=target_column)
        else:
            sns.pairplot(data_sample)
        plt.show()
        
    def plot_boxplots(self, data, target_column, figsize=(15, 8)):
        """
        Plot boxplots for numerical features by target.
        
        Args:
            data (pd.DataFrame): Input data
            target_column (str): Target column
            figsize (tuple): Figure size
        """
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        numerical_cols = numerical_cols.drop(target_column)
        
        n_cols = 3
        n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_rows > 1 else [axes]
        
        for i, col in enumerate(numerical_cols):
            if i < len(axes):
                sns.boxplot(data=data, x=target_column, y=col, ax=axes[i])
                axes[i].set_title(f'{col} by {target_column}')
                
        # Hide empty subplots
        for i in range(len(numerical_cols), len(axes)):
            axes[i].set_visible(False)
            
        plt.tight_layout()
        plt.show()
        
    def plot_missing_values(self, data, figsize=(12, 8)):
        """
        Plot missing values heatmap.
        
        Args:
            data (pd.DataFrame): Input data
            figsize (tuple): Figure size
        """
        plt.figure(figsize=figsize)
        sns.heatmap(data.isnull(), cbar=True, yticklabels=False)
        plt.title('Missing Values Heatmap')
        plt.show()
        
    def plot_target_distribution(self, data, target_column, figsize=(10, 6)):
        """
        Plot target variable distribution.
        
        Args:
            data (pd.DataFrame): Input data
            target_column (str): Target column
            figsize (tuple): Figure size
        """
        plt.figure(figsize=figsize)
        
        if data[target_column].dtype in ['object', 'category']:
            # Categorical target
            data[target_column].value_counts().plot(kind='bar')
            plt.title(f'Distribution of {target_column}')
            plt.xlabel(target_column)
            plt.ylabel('Count')
            plt.xticks(rotation=45)
        else:
            # Numerical target
            data[target_column].hist(bins=30, alpha=0.7)
            plt.title(f'Distribution of {target_column}')
            plt.xlabel(target_column)
            plt.ylabel('Frequency')
            
        plt.show()
        
    def plot_feature_importance(self, feature_names, importance_scores, 
                              top_n=20, figsize=(10, 8)):
        """
        Plot feature importance.
        
        Args:
            feature_names (list): Feature names
            importance_scores (array): Importance scores
            top_n (int): Number of top features to show
            figsize (tuple): Figure size
        """
        # Sort features by importance
        indices = np.argsort(importance_scores)[::-1][:top_n]
        
        plt.figure(figsize=figsize)
        plt.bar(range(top_n), importance_scores[indices])
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.title(f'Top {top_n} Feature Importance')
        plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=45)
        plt.tight_layout()
        plt.show()
        
    def plot_2d_projection(self, X, y=None, method='pca', figsize=(10, 8)):
        """
        Plot 2D projection of high-dimensional data.
        
        Args:
            X (array-like): Input data
            y (array-like): Target labels (optional)
            method (str): Projection method ('pca', 'tsne')
            figsize (tuple): Figure size
        """
        if method == 'pca':
            reducer = PCA(n_components=2)
            X_proj = reducer.fit_transform(X)
            title = 'PCA Projection'
        elif method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
            X_proj = reducer.fit_transform(X)
            title = 't-SNE Projection'
        else:
            raise ValueError("Method must be 'pca' or 'tsne'")
            
        plt.figure(figsize=figsize)
        
        if y is not None:
            scatter = plt.scatter(X_proj[:, 0], X_proj[:, 1], c=y, cmap='viridis', alpha=0.7)
            plt.colorbar(scatter)
        else:
            plt.scatter(X_proj[:, 0], X_proj[:, 1], alpha=0.7)
            
        plt.xlabel(f'{method.upper()} Component 1')
        plt.ylabel(f'{method.upper()} Component 2')
        plt.title(title)
        plt.show()


class ModelVisualizer:
    """
    Utility class for model visualization and evaluation.
    """
    
    def __init__(self):
        """
        Initialize ModelVisualizer.
        """
        pass
        
    def plot_confusion_matrix(self, y_true, y_pred, class_names=None, 
                            title="Confusion Matrix", figsize=(8, 6)):
        """
        Plot confusion matrix.
        
        Args:
            y_true (array-like): True labels
            y_pred (array-like): Predicted labels
            class_names (list): Names of classes
            title (str): Plot title
            figsize (tuple): Figure size
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
        
    def plot_roc_curve(self, y_true, y_proba, class_names=None, 
                      title="ROC Curve", figsize=(8, 6)):
        """
        Plot ROC curve.
        
        Args:
            y_true (array-like): True labels
            y_proba (array-like): Predicted probabilities
            class_names (list): Names of classes
            title (str): Plot title
            figsize (tuple): Figure size
        """
        from sklearn.metrics import roc_curve, roc_auc_score
        
        if len(np.unique(y_true)) == 2:
            # Binary classification
            fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
            auc_score = roc_auc_score(y_true, y_proba[:, 1])
            
            plt.figure(figsize=figsize)
            plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.2f})')
            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(title)
            plt.legend()
            plt.grid(True)
            plt.show()
        else:
            # Multi-class classification
            from sklearn.preprocessing import label_binarize
            
            y_bin = label_binarize(y_true, classes=np.unique(y_true))
            n_classes = y_bin.shape[1]
            
            plt.figure(figsize=figsize)
            
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
                auc_score = roc_auc_score(y_bin[:, i], y_proba[:, i])
                
                class_name = class_names[i] if class_names else f'Class {i}'
                plt.plot(fpr, tpr, label=f'{class_name} (AUC = {auc_score:.2f})')
                
            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(title)
            plt.legend()
            plt.grid(True)
            plt.show()
            
    def plot_learning_curves(self, train_scores, val_scores, 
                           title="Learning Curves", figsize=(10, 6)):
        """
        Plot learning curves.
        
        Args:
            train_scores (list): Training scores
            val_scores (list): Validation scores
            title (str): Plot title
            figsize (tuple): Figure size
        """
        epochs = range(1, len(train_scores) + 1)
        
        plt.figure(figsize=figsize)
        plt.plot(epochs, train_scores, 'b-', label='Training')
        plt.plot(epochs, val_scores, 'r-', label='Validation')
        plt.xlabel('Epochs')
        plt.ylabel('Score')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def plot_model_comparison(self, results_dict, metric='accuracy', 
                            title="Model Comparison", figsize=(10, 6)):
        """
        Plot model comparison bar chart.
        
        Args:
            results_dict (dict): Dictionary with model names and metrics
            metric (str): Metric to compare
            title (str): Plot title
            figsize (tuple): Figure size
        """
        models = list(results_dict.keys())
        scores = [results_dict[model].get(metric, 0) for model in models]
        
        plt.figure(figsize=figsize)
        bars = plt.bar(models, scores)
        plt.xlabel('Models')
        plt.ylabel(metric.capitalize())
        plt.title(title)
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
                    
        plt.tight_layout()
        plt.show()
