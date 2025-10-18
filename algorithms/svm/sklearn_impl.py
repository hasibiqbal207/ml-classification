"""
Support Vector Machine (SVM) Algorithm Implementation

SVM is a powerful classification algorithm that finds the optimal hyperplane
to separate classes with maximum margin.
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class SVMSklearn:
    """
    Support Vector Machine implementation using Scikit-learn.
    """
    
    def __init__(self, C=1.0, kernel='rbf', gamma='scale', random_state=42):
        """
        Initialize SVM model.
        
        Args:
            C (float): Regularization parameter
            kernel (str): Kernel type ('linear', 'poly', 'rbf', 'sigmoid')
            gamma (str or float): Kernel coefficient
            random_state (int): Random state for reproducibility
        """
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.random_state = random_state
        
        self.model = SVC(
            C=self.C,
            kernel=self.kernel,
            gamma=self.gamma,
            random_state=self.random_state,
            probability=True  # Enable probability estimates
        )
        
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def fit(self, X, y):
        """
        Train the SVM model.
        
        Args:
            X (array-like): Training features
            y (array-like): Training labels
            
        Returns:
            self: Fitted model
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        return self
        
    def predict(self, X):
        """
        Make predictions.
        
        Args:
            X (array-like): Features to predict
            
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
            X (array-like): Features to predict
            
        Returns:
            array: Class probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
        
    def score(self, X, y):
        """
        Calculate accuracy score.
        
        Args:
            X (array-like): Features
            y (array-like): True labels
            
        Returns:
            float: Accuracy score
        """
        predictions = self.predict(X)
        return accuracy_score(y, predictions)
        
    def get_support_vectors(self):
        """
        Get support vectors.
        
        Returns:
            array: Support vectors
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before accessing support vectors")
            
        return self.model.support_vectors_
        
    def get_decision_function(self, X):
        """
        Get decision function values.
        
        Args:
            X (array-like): Features
            
        Returns:
            array: Decision function values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before accessing decision function")
            
        X_scaled = self.scaler.transform(X)
        return self.model.decision_function(X_scaled)
        
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model.
        
        Args:
            X_test (array-like): Test features
            y_test (array-like): Test labels
            
        Returns:
            dict: Evaluation metrics
        """
        predictions = self.predict(X_test)
        probabilities = self.predict_proba(X_test)
        
        accuracy = accuracy_score(y_test, predictions)
        
        # Get classification report
        report = classification_report(y_test, predictions, output_dict=True)
        
        # Get confusion matrix
        cm = confusion_matrix(y_test, predictions)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': predictions,
            'probabilities': probabilities
        }
        
    def plot_decision_boundary(self, X, y, title="SVM Decision Boundary"):
        """
        Plot decision boundary (for 2D data).
        
        Args:
            X (array-like): Features (2D)
            y (array-like): Labels
            title (str): Plot title
        """
        if X.shape[1] != 2:
            raise ValueError("Decision boundary plotting only supports 2D features")
            
        # Create a mesh
        h = 0.02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        # Make predictions on the mesh
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        Z = self.predict(mesh_points)
        Z = Z.reshape(xx.shape)
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
        scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='black')
        plt.colorbar(scatter)
        plt.title(title)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.show()
        
    def plot_support_vectors(self, X, y, title="SVM Support Vectors"):
        """
        Plot support vectors.
        
        Args:
            X (array-like): Features
            y (array-like): Labels
            title (str): Plot title
        """
        if X.shape[1] != 2:
            raise ValueError("Support vector plotting only supports 2D features")
            
        support_vectors = self.get_support_vectors()
        
        plt.figure(figsize=(10, 8))
        
        # Plot all points
        scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, alpha=0.6)
        
        # Highlight support vectors
        plt.scatter(support_vectors[:, 0], support_vectors[:, 1], 
                   s=100, facecolors='none', edgecolors='black', linewidth=2)
        
        plt.colorbar(scatter)
        plt.title(title)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.show()
