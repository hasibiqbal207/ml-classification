# K-Means Clustering Algorithm

## Overview
K-Means is an unsupervised learning algorithm that partitions data into k clusters, where each data point belongs to the cluster with the nearest mean (centroid). It's one of the most popular clustering algorithms due to its simplicity and effectiveness.

## Algorithm Steps

### 1. Initialization
- Choose k initial centroids (cluster centers)
- Common methods: random initialization, k-means++

### 2. Assignment
- Assign each data point to the nearest centroid
- Distance metric: usually Euclidean distance

### 3. Update
- Recalculate centroids as the mean of assigned points
- Update cluster centers

### 4. Convergence
- Repeat steps 2-3 until convergence
- Stop when centroids no longer change significantly

## Mathematical Foundation

### Objective Function
```
J = Σ(i=1 to k) Σ(x∈C_i) ||x - μ_i||²
```
where:
- C_i is cluster i
- μ_i is centroid of cluster i
- ||x - μ_i||² is squared Euclidean distance

### Centroid Update
```
μ_i = (1/|C_i|) Σ(x∈C_i) x
```

### Assignment Rule
```
c_i = argmin_j ||x_i - μ_j||²
```

## Key Concepts

### Distance Metrics
- **Euclidean Distance**: Most common for continuous features
- **Manhattan Distance**: Alternative for high-dimensional data
- **Cosine Similarity**: Useful for text data

### Initialization Methods
- **Random**: Randomly select k points as initial centroids
- **K-means++**: Smart initialization to avoid poor local minima
- **K-means||**: Scalable version for large datasets

### Convergence Criteria
- **Centroid Stability**: Centroids don't change between iterations
- **Maximum Iterations**: Stop after specified number of iterations
- **Tolerance**: Stop when change is below threshold

## Advantages
- **Simple and Fast**: Easy to understand and implement
- **Scalable**: Works well with large datasets
- **Versatile**: Can be applied to various data types
- **Deterministic**: Same results with same initialization
- **Memory Efficient**: Low memory requirements

## Disadvantages
- **Requires k**: Must specify number of clusters beforehand
- **Sensitive to Initialization**: Different initializations can yield different results
- **Assumes Spherical Clusters**: Works best with circular/spherical clusters
- **Sensitive to Outliers**: Outliers can significantly affect centroids
- **Local Minima**: May get stuck in suboptimal solutions

## Evaluation Metrics

### Silhouette Score
- Measures how similar an object is to its own cluster vs other clusters
- Range: [-1, 1], higher is better

### Calinski-Harabasz Index
- Ratio of between-cluster to within-cluster dispersion
- Higher values indicate better clustering

### Davies-Bouldin Index
- Average similarity ratio of clusters
- Lower values indicate better clustering

### Inertia (Within-cluster Sum of Squares)
- Sum of squared distances to centroids
- Lower values indicate tighter clusters

## Hyperparameters
- **n_clusters (k)**: Number of clusters to form
- **max_iter**: Maximum number of iterations
- **init**: Initialization method
- **n_init**: Number of times algorithm is run
- **random_state**: Random seed for reproducibility

## Implementation Notes
- Scikit-learn implementation includes:
  - Automatic feature scaling
  - Multiple evaluation metrics
  - Visualization capabilities
  - Optimal cluster number finding
- Data should be normalized for best results
- K-means++ initialization is recommended
- Use multiple random initializations to avoid local minima

## Applications
- **Customer Segmentation**: Group customers by behavior
- **Image Segmentation**: Partition images into regions
- **Document Clustering**: Group similar documents
- **Gene Expression Analysis**: Cluster genes with similar expression
- **Market Research**: Segment markets and products

## References
- MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations
- Arthur, D., & Vassilvitskii, S. (2007). K-means++: The advantages of careful seeding
- Lloyd, S. (1982). Least squares quantization in PCM
- Scikit-learn Documentation: https://scikit-learn.org/stable/modules/clustering.html#k-means
