"""
Logistic Regression Implementation using Scikit-learn
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class LogisticRegressionSklearn:
    """
    Logistic Regression classifier using Scikit-learn implementation.
    
    This class provides a wrapper around sklearn's LogisticRegression with
    additional functionality for preprocessing and evaluation.
    """
    
    def __init__(self, random_state=42, max_iter=1000):
        """
        Initialize the Logistic Regression model.
        
        Args:
            random_state (int): Random state for reproducibility
            max_iter (int): Maximum number of iterations for convergence
        """
        self.model = LogisticRegression(random_state=random_state, max_iter=max_iter)
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def fit(self, X, y):
        """
        Fit the logistic regression model.
        
        Args:
            X (array-like): Training features
            y (array-like): Training labels
        """
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit the model
        self.model.fit(X_scaled, y)
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
            
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
        
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
            
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
        
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
