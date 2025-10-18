"""
K-Nearest Neighbors (KNN) Algorithm Implementation

KNN is a simple, non-parametric classification algorithm that classifies
data points based on the majority class of their k nearest neighbors.
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class KNNSklearn:
    """
    K-Nearest Neighbors implementation using Scikit-learn.
    """
    
    def __init__(self, n_neighbors=5, weights='uniform', algorithm='auto', 
                 leaf_size=30, p=2, metric='minkowski', random_state=42):
        """
        Initialize KNN model.
        
        Args:
            n_neighbors (int): Number of neighbors to use
            weights (str): Weight function ('uniform', 'distance')
            algorithm (str): Algorithm to use ('auto', 'ball_tree', 'kd_tree', 'brute')
            leaf_size (int): Leaf size for tree algorithms
            p (int): Power parameter for Minkowski metric
            metric (str): Distance metric to use
            random_state (int): Random state for reproducibility
        """
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.metric = metric
        self.random_state = random_state
        
        self.model = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            p=self.p,
            metric=self.metric
        )
        
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def fit(self, X, y):
        """
        Train the KNN model.
        
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
        
    def kneighbors(self, X, n_neighbors=None, return_distance=True):
        """
        Find the k-neighbors of a point.
        
        Args:
            X (array-like): Query points
            n_neighbors (int): Number of neighbors to find
            return_distance (bool): Whether to return distances
            
        Returns:
            tuple: (distances, indices) if return_distance=True, else indices
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before finding neighbors")
            
        X_scaled = self.scaler.transform(X)
        
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
            
        return self.model.kneighbors(X_scaled, n_neighbors, return_distance)
        
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
        
    def plot_decision_boundary(self, X, y, title="KNN Decision Boundary"):
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
        
    def plot_k_vs_accuracy(self, X_train, y_train, X_test, y_test, k_range=None, title="KNN: K vs Accuracy"):
        """
        Plot accuracy vs number of neighbors.
        
        Args:
            X_train (array-like): Training features
            y_train (array-like): Training labels
            X_test (array-like): Test features
            y_test (array-like): Test labels
            k_range (range): Range of k values to test
            title (str): Plot title
        """
        if k_range is None:
            k_range = range(1, 21)
            
        train_scores = []
        test_scores = []
        
        for k in k_range:
            # Create temporary model
            temp_model = KNeighborsClassifier(n_neighbors=k, weights=self.weights)
            temp_scaler = StandardScaler()
            
            # Scale and fit
            X_train_scaled = temp_scaler.fit_transform(X_train)
            X_test_scaled = temp_scaler.transform(X_test)
            
            temp_model.fit(X_train_scaled, y_train)
            
            # Calculate scores
            train_score = temp_model.score(X_train_scaled, y_train)
            test_score = temp_model.score(X_test_scaled, y_test)
            
            train_scores.append(train_score)
            test_scores.append(test_score)
            
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(k_range, train_scores, 'o-', label='Training Accuracy')
        plt.plot(k_range, test_scores, 'o-', label='Test Accuracy')
        plt.xlabel('Number of Neighbors (K)')
        plt.ylabel('Accuracy')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
    def plot_distance_weights(self, X, y, title="KNN Distance Weights"):
        """
        Plot how distance weights affect predictions.
        
        Args:
            X (array-like): Features
            y (array-like): Labels
            title (str): Plot title
        """
        if X.shape[1] != 2:
            raise ValueError("Distance weights plotting only supports 2D features")
            
        # Create a mesh
        h = 0.02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        # Compare uniform vs distance weights
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        for i, weights in enumerate(['uniform', 'distance']):
            temp_model = KNeighborsClassifier(n_neighbors=self.n_neighbors, weights=weights)
            temp_scaler = StandardScaler()
            
            X_scaled = temp_scaler.fit_transform(X)
            temp_model.fit(X_scaled, y)
            
            # Make predictions on the mesh
            mesh_points = np.c_[xx.ravel(), yy.ravel()]
            mesh_scaled = temp_scaler.transform(mesh_points)
            Z = temp_model.predict(mesh_scaled)
            Z = Z.reshape(xx.shape)
            
            # Plot
            ax = ax1 if i == 0 else ax2
            ax.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
            scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='black')
            ax.set_title(f'KNN with {weights} weights')
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
        
    def get_neighbor_distances(self, X):
        """
        Get distances to nearest neighbors.
        
        Args:
            X (array-like): Query points
            
        Returns:
            array: Distances to nearest neighbors
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting neighbor distances")
            
        distances, _ = self.kneighbors(X, return_distance=True)
        return distances
        
    def get_neighbor_indices(self, X):
        """
        Get indices of nearest neighbors.
        
        Args:
            X (array-like): Query points
            
        Returns:
            array: Indices of nearest neighbors
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting neighbor indices")
            
        _, indices = self.kneighbors(X, return_distance=True)
        return indices
