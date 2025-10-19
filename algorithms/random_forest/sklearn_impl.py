"""
Random Forest Algorithm Implementation

Random Forest is an ensemble learning method that combines multiple decision trees
to create a more robust and accurate classifier.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class RandomForestSklearn:
    """
    Random Forest implementation using Scikit-learn.
    """
    
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, 
                 min_samples_leaf=1, random_state=42, n_jobs=-1):
        """
        Initialize Random Forest model.
        
        Args:
            n_estimators (int): Number of trees in the forest
            max_depth (int): Maximum depth of trees
            min_samples_split (int): Minimum samples required to split a node
            min_samples_leaf (int): Minimum samples required at a leaf node
            random_state (int): Random state for reproducibility
            n_jobs (int): Number of jobs to run in parallel
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
        
        self.is_fitted = False
        
    def fit(self, X, y):
        """
        Train the Random Forest model.
        
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
        
    def get_feature_importance(self):
        """
        Get feature importance scores.
        
        Returns:
            array: Feature importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before accessing feature importance")
            
        return self.model.feature_importances_
        
    def get_tree_depth(self):
        """
        Get the depth of each tree in the forest.
        
        Returns:
            array: Depth of each tree
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before accessing tree depth")
            
        return [tree.get_depth() for tree in self.model.estimators_]
        
    def get_tree_leaves(self):
        """
        Get the number of leaves in each tree.
        
        Returns:
            array: Number of leaves in each tree
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before accessing tree leaves")
            
        return [tree.get_n_leaves() for tree in self.model.estimators_]
        
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
        
    def plot_feature_importance(self, feature_names=None, top_n=20, title="Random Forest Feature Importance", save_path=None):
        """
        Plot feature importance.
        
        Args:
            feature_names (list): Names of features
            top_n (int): Number of top features to show
            title (str): Plot title
            save_path (str): Path to save the plot (optional)
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
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        
    def plot_tree_depth_distribution(self, title="Random Forest Tree Depth Distribution"):
        """
        Plot distribution of tree depths.
        
        Args:
            title (str): Plot title
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before plotting tree depth")
            
        depths = self.get_tree_depth()
        
        plt.figure(figsize=(10, 6))
        plt.hist(depths, bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Tree Depth')
        plt.ylabel('Frequency')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.show()
        
    def plot_oob_error(self, title="Random Forest OOB Error"):
        """
        Plot out-of-bag error vs number of trees.
        
        Args:
            title (str): Plot title
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before plotting OOB error")
            
        # Get OOB scores for each tree
        oob_scores = []
        for i in range(1, self.n_estimators + 1):
            # Create a temporary forest with i trees
            temp_forest = RandomForestClassifier(
                n_estimators=i,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                oob_score=True
            )
            temp_forest.fit(self.model.estimators_[0].tree_.X, self.model.estimators_[0].tree_.y)
            oob_scores.append(1 - temp_forest.oob_score_)
            
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, self.n_estimators + 1), oob_scores)
        plt.xlabel('Number of Trees')
        plt.ylabel('OOB Error')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.show()
        
    def get_individual_tree_predictions(self, X):
        """
        Get predictions from individual trees.
        
        Args:
            X (array-like): Features
            
        Returns:
            array: Predictions from each tree
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting individual predictions")
            
        predictions = []
        for tree in self.model.estimators_:
            predictions.append(tree.predict(X))
            
        return np.array(predictions)
        
    def get_prediction_confidence(self, X):
        """
        Get prediction confidence (variance across trees).
        
        Args:
            X (array-like): Features
            
        Returns:
            array: Prediction confidence scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting prediction confidence")
            
        individual_predictions = self.get_individual_tree_predictions(X)
        
        # Calculate variance across trees for each sample
        confidence = np.var(individual_predictions, axis=0)
        
        return confidence
