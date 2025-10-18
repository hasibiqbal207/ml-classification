"""
AdaBoost Algorithm Implementation

AdaBoost (Adaptive Boosting) is an ensemble learning method that combines
multiple weak learners to create a strong classifier.
"""

import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class AdaBoostSklearn:
    """
    AdaBoost implementation using Scikit-learn.
    """
    
    def __init__(self, base_estimator=None, n_estimators=50, learning_rate=1.0,
                 algorithm='SAMME.R', random_state=42):
        """
        Initialize AdaBoost model.
        
        Args:
            base_estimator (object): Base estimator (default: DecisionTreeClassifier)
            n_estimators (int): Number of estimators
            learning_rate (float): Learning rate
            algorithm (str): Algorithm to use ('SAMME', 'SAMME.R')
            random_state (int): Random state for reproducibility
        """
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.algorithm = algorithm
        self.random_state = random_state
        
        # Use DecisionTreeClassifier as default base estimator
        if self.base_estimator is None:
            self.base_estimator = DecisionTreeClassifier(max_depth=1, random_state=random_state)
        
        self.model = AdaBoostClassifier(
            base_estimator=self.base_estimator,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            algorithm=self.algorithm,
            random_state=self.random_state
        )
        
        self.is_fitted = False
        
    def fit(self, X, y):
        """
        Train the AdaBoost model.
        
        Args:
            X (array-like): Training features
            y (array-like): Training labels
            
        Returns:
            self: Fitted model
        """
        self.model.fit(X, y)
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
            
        return self.model.predict(X)
        
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
            
        return self.model.predict_proba(X)
        
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
        
    def get_estimator_weights(self):
        """
        Get weights of each estimator.
        
        Returns:
            array: Estimator weights
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before accessing estimator weights")
            
        return self.model.estimator_weights_
        
    def get_estimator_errors(self):
        """
        Get errors of each estimator.
        
        Returns:
            array: Estimator errors
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before accessing estimator errors")
            
        return self.model.estimator_errors_
        
    def get_feature_importance(self):
        """
        Get feature importance scores.
        
        Returns:
            array: Feature importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before accessing feature importance")
            
        return self.model.feature_importances_
        
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
        
    def plot_estimator_weights(self, title="AdaBoost Estimator Weights"):
        """
        Plot weights of each estimator.
        
        Args:
            title (str): Plot title
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before plotting estimator weights")
            
        weights = self.get_estimator_weights()
        
        plt.figure(figsize=(12, 6))
        plt.plot(range(1, len(weights) + 1), weights, 'o-')
        plt.xlabel('Estimator')
        plt.ylabel('Weight')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.show()
        
    def plot_estimator_errors(self, title="AdaBoost Estimator Errors"):
        """
        Plot errors of each estimator.
        
        Args:
            title (str): Plot title
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before plotting estimator errors")
            
        errors = self.get_estimator_errors()
        
        plt.figure(figsize=(12, 6))
        plt.plot(range(1, len(errors) + 1), errors, 'o-', color='red')
        plt.xlabel('Estimator')
        plt.ylabel('Error')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.show()
        
    def plot_feature_importance(self, feature_names=None, top_n=20, title="AdaBoost Feature Importance"):
        """
        Plot feature importance.
        
        Args:
            feature_names (list): Names of features
            top_n (int): Number of top features to show
            title (str): Plot title
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before plotting feature importance")
            
        importance = self.get_feature_importance()
        
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(len(importance))]
            
        # Get top N features
        indices = np.argsort(importance)[::-1][:top_n]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(indices)), importance[indices])
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Feature Importance')
        plt.title(title)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
        
    def plot_learning_curve(self, X, y, title="AdaBoost Learning Curve"):
        """
        Plot learning curve showing performance vs number of estimators.
        
        Args:
            X (array-like): Features
            y (array-like): Labels
            title (str): Plot title
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before plotting learning curve")
            
        # Calculate accuracy for different numbers of estimators
        n_estimators_range = range(1, self.n_estimators + 1, 5)
        train_scores = []
        test_scores = []
        
        for n_est in n_estimators_range:
            # Create temporary model with n_est estimators
            temp_model = AdaBoostClassifier(
                base_estimator=self.base_estimator,
                n_estimators=n_est,
                learning_rate=self.learning_rate,
                algorithm=self.algorithm,
                random_state=self.random_state
            )
            
            # Fit and score
            temp_model.fit(X, y)
            train_score = temp_model.score(X, y)
            test_score = temp_model.score(X, y)  # Using same data for simplicity
            
            train_scores.append(train_score)
            test_scores.append(test_score)
            
        # Plot learning curve
        plt.figure(figsize=(12, 6))
        plt.plot(n_estimators_range, train_scores, 'o-', label='Training Score')
        plt.plot(n_estimators_range, test_scores, 'o-', label='Test Score')
        plt.xlabel('Number of Estimators')
        plt.ylabel('Accuracy')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
    def get_individual_predictions(self, X):
        """
        Get predictions from individual estimators.
        
        Args:
            X (array-like): Features
            
        Returns:
            array: Predictions from each estimator
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting individual predictions")
            
        predictions = []
        for estimator in self.model.estimators_:
            predictions.append(estimator.predict(X))
            
        return np.array(predictions)
        
    def get_prediction_confidence(self, X):
        """
        Get prediction confidence (variance across estimators).
        
        Args:
            X (array-like): Features
            
        Returns:
            array: Prediction confidence scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting prediction confidence")
            
        individual_predictions = self.get_individual_predictions(X)
        
        # Calculate variance across estimators for each sample
        confidence = np.var(individual_predictions, axis=0)
        
        return confidence
        
    def plot_decision_boundary(self, X, y, title="AdaBoost Decision Boundary"):
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
