"""
Decision Tree Implementation using Scikit-learn
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import export_text, plot_tree
import matplotlib.pyplot as plt


class DecisionTreeSklearn:
    """
    Decision Tree classifier using Scikit-learn implementation.
    
    This class provides a wrapper around sklearn's DecisionTreeClassifier with
    additional functionality for visualization and evaluation.
    """
    
    def __init__(self, random_state=42, max_depth=None, min_samples_split=2, 
                 min_samples_leaf=1, criterion='gini'):
        """
        Initialize the Decision Tree model.
        
        Args:
            random_state (int): Random state for reproducibility
            max_depth (int): Maximum depth of the tree
            min_samples_split (int): Minimum samples required to split a node
            min_samples_leaf (int): Minimum samples required at a leaf node
            criterion (str): Function to measure quality of split ('gini' or 'entropy')
        """
        self.model = DecisionTreeClassifier(
            random_state=random_state,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            criterion=criterion
        )
        self.is_fitted = False
        
    def fit(self, X, y):
        """
        Fit the decision tree model.
        
        Args:
            X (array-like): Training features
            y (array-like): Training labels
        """
        self.model.fit(X, y)
        self.is_fitted = True
        
    def predict(self, X):
        """
        Make predictions on new data.
        
        Args:
            X (array-like): Features to predict on
            
        Returns:
            array: Predicted labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        return self.model.predict(X)
        
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Args:
            X (array-like): Features to predict on
            
        Returns:
            array: Class probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        return self.model.predict_proba(X)
        
    def get_feature_importance(self):
        """
        Get feature importance scores.
        
        Returns:
            array: Feature importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
            
        return self.model.feature_importances_
        
    def visualize_tree(self, feature_names=None, class_names=None, max_depth=3):
        """
        Visualize the decision tree.
        
        Args:
            feature_names (list): Names of features
            class_names (list): Names of classes
            max_depth (int): Maximum depth to visualize
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before visualizing")
            
        plt.figure(figsize=(20, 10))
        plot_tree(
            self.model,
            feature_names=feature_names,
            class_names=class_names,
            max_depth=max_depth,
            filled=True,
            rounded=True,
            fontsize=10
        )
        plt.title("Decision Tree Visualization")
        plt.show()
        
    def get_tree_text(self, feature_names=None, max_depth=3):
        """
        Get text representation of the decision tree.
        
        Args:
            feature_names (list): Names of features
            max_depth (int): Maximum depth to show
            
        Returns:
            str: Text representation of the tree
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting tree text")
            
        return export_text(
            self.model,
            feature_names=feature_names,
            max_depth=max_depth
        )
        
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model performance.
        
        Args:
            X_test (array-like): Test features
            y_test (array-like): Test labels
            
        Returns:
            dict: Evaluation metrics
        """
        y_pred = self.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        return {
            'accuracy': accuracy,
            'predictions': y_pred,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
