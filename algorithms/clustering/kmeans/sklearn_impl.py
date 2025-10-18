"""
K-Means Clustering Implementation using Scikit-learn
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


class KMeansSklearn:
    """
    K-Means clustering using Scikit-learn implementation.
    
    This class provides a wrapper around sklearn's KMeans with
    additional functionality for evaluation and visualization.
    """
    
    def __init__(self, n_clusters=8, random_state=42, max_iter=300, 
                 init='k-means++', n_init=10):
        """
        Initialize the K-Means model.
        
        Args:
            n_clusters (int): Number of clusters to form
            random_state (int): Random state for reproducibility
            max_iter (int): Maximum number of iterations
            init (str): Initialization method ('k-means++', 'random')
            n_init (int): Number of times algorithm is run with different centroids
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.max_iter = max_iter
        self.init = init
        self.n_init = n_init
        
        self.model = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            max_iter=max_iter,
            init=init,
            n_init=n_init
        )
        
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def fit(self, X):
        """
        Fit the K-Means model.
        
        Args:
            X (array-like): Training data
        """
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit the model
        self.model.fit(X_scaled)
        self.is_fitted = True
        
    def predict(self, X):
        """
        Predict cluster labels for new data.
        
        Args:
            X (array-like): Data to cluster
            
        Returns:
            array: Cluster labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
        
    def fit_predict(self, X):
        """
        Fit the model and predict cluster labels.
        
        Args:
            X (array-like): Data to cluster
            
        Returns:
            array: Cluster labels
        """
        self.fit(X)
        return self.predict(X)
        
    def get_centroids(self):
        """
        Get cluster centroids.
        
        Returns:
            array: Cluster centroids
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting centroids")
            
        return self.model.cluster_centers_
        
    def get_inertia(self):
        """
        Get sum of squared distances of samples to their closest cluster center.
        
        Returns:
            float: Inertia value
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting inertia")
            
        return self.model.inertia_
        
    def evaluate(self, X):
        """
        Evaluate clustering performance using multiple metrics.
        
        Args:
            X (array-like): Data used for evaluation
            
        Returns:
            dict: Evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
            
        X_scaled = self.scaler.transform(X)
        labels = self.model.predict(X_scaled)
        
        # Calculate metrics
        silhouette = silhouette_score(X_scaled, labels)
        calinski_harabasz = calinski_harabasz_score(X_scaled, labels)
        davies_bouldin = davies_bouldin_score(X_scaled, labels)
        inertia = self.model.inertia_
        
        print(f"Number of clusters: {self.n_clusters}")
        print(f"Silhouette Score: {silhouette:.4f}")
        print(f"Calinski-Harabasz Score: {calinski_harabasz:.4f}")
        print(f"Davies-Bouldin Score: {davies_bouldin:.4f}")
        print(f"Inertia: {inertia:.4f}")
        
        return {
            'silhouette_score': silhouette,
            'calinski_harabasz_score': calinski_harabasz,
            'davies_bouldin_score': davies_bouldin,
            'inertia': inertia,
            'labels': labels
        }
        
    def plot_clusters(self, X, feature_names=None, title="K-Means Clustering"):
        """
        Plot clusters (works for 2D data).
        
        Args:
            X (array-like): Data to plot
            feature_names (list): Names of features
            title (str): Plot title
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before plotting")
            
        X_scaled = self.scaler.transform(X)
        labels = self.model.predict(X_scaled)
        centroids = self.model.cluster_centers_
        
        if X_scaled.shape[1] != 2:
            print("Plotting is only supported for 2D data")
            return
            
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis', alpha=0.7)
        plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, linewidths=3)
        
        if feature_names:
            plt.xlabel(feature_names[0])
            plt.ylabel(feature_names[1])
        else:
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            
        plt.title(title)
        plt.colorbar(scatter)
        plt.show()
        
    def find_optimal_clusters(self, X, max_clusters=10):
        """
        Find optimal number of clusters using elbow method and silhouette analysis.
        
        Args:
            X (array-like): Data to analyze
            max_clusters (int): Maximum number of clusters to test
            
        Returns:
            dict: Results of cluster analysis
        """
        X_scaled = self.scaler.fit_transform(X)
        
        inertias = []
        silhouette_scores = []
        k_range = range(2, max_clusters + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state)
            labels = kmeans.fit_predict(X_scaled)
            
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X_scaled, labels))
            
        # Plot results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Elbow method
        ax1.plot(k_range, inertias, 'bo-')
        ax1.set_xlabel('Number of Clusters')
        ax1.set_ylabel('Inertia')
        ax1.set_title('Elbow Method')
        ax1.grid(True)
        
        # Silhouette analysis
        ax2.plot(k_range, silhouette_scores, 'ro-')
        ax2.set_xlabel('Number of Clusters')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Analysis')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Find optimal k
        optimal_k_silhouette = k_range[np.argmax(silhouette_scores)]
        
        return {
            'k_range': list(k_range),
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'optimal_k_silhouette': optimal_k_silhouette
        }
