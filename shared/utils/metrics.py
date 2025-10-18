"""
Metrics utilities for evaluating model performance.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns


class ClassificationMetrics:
    """
    Utility class for classification metrics and visualization.
    """
    
    def __init__(self):
        """
        Initialize ClassificationMetrics.
        """
        pass
        
    def calculate_metrics(self, y_true, y_pred, y_proba=None, average='weighted'):
        """
        Calculate comprehensive classification metrics.
        
        Args:
            y_true (array-like): True labels
            y_pred (array-like): Predicted labels
            y_proba (array-like): Predicted probabilities (optional)
            average (str): Averaging method for multi-class metrics
            
        Returns:
            dict: Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average=average),
            'recall': recall_score(y_true, y_pred, average=average),
            'f1_score': f1_score(y_true, y_pred, average=average)
        }
        
        # Add AUC score if probabilities are provided
        if y_proba is not None:
            try:
                if len(np.unique(y_true)) == 2:
                    # Binary classification
                    metrics['auc'] = roc_auc_score(y_true, y_proba[:, 1])
                else:
                    # Multi-class classification
                    metrics['auc'] = roc_auc_score(y_true, y_proba, multi_class='ovr')
            except:
                metrics['auc'] = None
                
        return metrics
        
    def print_metrics(self, y_true, y_pred, y_proba=None):
        """
        Print formatted classification metrics.
        
        Args:
            y_true (array-like): True labels
            y_pred (array-like): Predicted labels
            y_proba (array-like): Predicted probabilities (optional)
        """
        metrics = self.calculate_metrics(y_true, y_pred, y_proba)
        
        print("Classification Metrics:")
        print("-" * 30)
        for metric, value in metrics.items():
            if value is not None:
                print(f"{metric.capitalize()}: {value:.4f}")
                
        print("\nDetailed Classification Report:")
        print("-" * 40)
        print(classification_report(y_true, y_pred))
        
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
            
            # Binarize the output
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
            
    def plot_precision_recall_curve(self, y_true, y_proba, 
                                  title="Precision-Recall Curve", figsize=(8, 6)):
        """
        Plot precision-recall curve.
        
        Args:
            y_true (array-like): True labels
            y_proba (array-like): Predicted probabilities
            title (str): Plot title
            figsize (tuple): Figure size
        """
        if len(np.unique(y_true)) == 2:
            # Binary classification
            precision, recall, _ = precision_recall_curve(y_true, y_proba[:, 1])
            avg_precision = average_precision_score(y_true, y_proba[:, 1])
            
            plt.figure(figsize=figsize)
            plt.plot(recall, precision, label=f'PR Curve (AP = {avg_precision:.2f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(title)
            plt.legend()
            plt.grid(True)
            plt.show()
        else:
            print("Precision-Recall curve is only supported for binary classification")
            
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


class ClusteringMetrics:
    """
    Utility class for clustering metrics and evaluation.
    """
    
    def __init__(self):
        """
        Initialize ClusteringMetrics.
        """
        pass
        
    def calculate_metrics(self, X, labels):
        """
        Calculate clustering metrics.
        
        Args:
            X (array-like): Input data
            labels (array-like): Cluster labels
            
        Returns:
            dict: Dictionary of metrics
        """
        from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
        
        metrics = {
            'silhouette_score': silhouette_score(X, labels),
            'calinski_harabasz_score': calinski_harabasz_score(X, labels),
            'davies_bouldin_score': davies_bouldin_score(X, labels)
        }
        
        return metrics
        
    def print_metrics(self, X, labels):
        """
        Print formatted clustering metrics.
        
        Args:
            X (array-like): Input data
            labels (array-like): Cluster labels
        """
        metrics = self.calculate_metrics(X, labels)
        
        print("Clustering Metrics:")
        print("-" * 30)
        for metric, value in metrics.items():
            print(f"{metric.replace('_', ' ').title()}: {value:.4f}")
            
    def plot_silhouette_analysis(self, X, labels, title="Silhouette Analysis", figsize=(10, 6)):
        """
        Plot silhouette analysis.
        
        Args:
            X (array-like): Input data
            labels (array-like): Cluster labels
            title (str): Plot title
            figsize (tuple): Figure size
        """
        from sklearn.metrics import silhouette_samples
        
        silhouette_vals = silhouette_samples(X, labels)
        y_lower = 10
        
        plt.figure(figsize=figsize)
        
        for i in range(len(np.unique(labels))):
            cluster_silhouette_vals = silhouette_vals[labels == i]
            cluster_silhouette_vals.sort()
            
            size_cluster_i = cluster_silhouette_vals.shape[0]
            y_upper = y_lower + size_cluster_i
            
            plt.fill_betweenx(np.arange(y_lower, y_upper),
                             0, cluster_silhouette_vals,
                             alpha=0.7)
            plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10
            
        plt.xlabel('Silhouette Coefficient')
        plt.ylabel('Cluster')
        plt.title(title)
        plt.show()


def compare_models(results_dict, metric='accuracy'):
    """
    Compare multiple models based on a specific metric.
    
    Args:
        results_dict (dict): Dictionary with model names as keys and metrics as values
        metric (str): Metric to compare
        
    Returns:
        pd.DataFrame: Comparison results
    """
    import pandas as pd
    
    comparison_data = []
    for model_name, metrics in results_dict.items():
        comparison_data.append({
            'Model': model_name,
            metric.capitalize(): metrics.get(metric, 0)
        })
        
    df = pd.DataFrame(comparison_data)
    df = df.sort_values(metric.capitalize(), ascending=False)
    
    return df
