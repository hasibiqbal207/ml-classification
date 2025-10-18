# K-Nearest Neighbors (KNN) Algorithm

## Overview
K-Nearest Neighbors is a simple, non-parametric classification algorithm that classifies data points based on the majority class of their k nearest neighbors in the feature space.

## Mathematical Foundation

### Distance Metrics
KNN uses various distance metrics to find nearest neighbors:

**Euclidean Distance:**
```
d(x, y) = √(Σ(x_i - y_i)²)
```

**Manhattan Distance:**
```
d(x, y) = Σ|x_i - y_i|
```

**Minkowski Distance:**
```
d(x, y) = (Σ|x_i - y_i|^p)^(1/p)
```

### Classification Rule
For a query point x, find k nearest neighbors and assign the majority class:

```
ŷ = argmax_c Σ I(y_i = c) for i ∈ N_k(x)
```

Where:
- `N_k(x)` is the set of k nearest neighbors
- `I(y_i = c)` is an indicator function
- `c` is a class label

### Weighted Voting
When using distance weights:

```
ŷ = argmax_c Σ w_i * I(y_i = c) for i ∈ N_k(x)
```

Where:
```
w_i = 1 / d(x, x_i) for distance weights
```

## Key Concepts
- **Lazy Learning**: No explicit training phase
- **Instance-Based**: Stores all training data
- **Distance Metrics**: Various ways to measure similarity
- **Weighted Voting**: Distance-based or uniform weights
- **Curse of Dimensionality**: Performance degrades with high dimensions

## Advantages
- Simple to understand and implement
- No assumptions about data distribution
- Works well for non-linear decision boundaries
- Naturally handles multi-class problems
- Can be used for both classification and regression

## Disadvantages
- Computationally expensive for large datasets
- Sensitive to irrelevant features
- Requires feature scaling
- Memory intensive (stores all training data)
- Sensitive to the choice of k
- Poor performance with high-dimensional data

## Implementation Notes
- Automatic feature scaling is included
- Supports multiple distance metrics
- Comprehensive visualization methods
- K-value optimization analysis
- Distance weight analysis

## Hyperparameters
- **n_neighbors**: Number of neighbors to use (default: 5)
- **weights**: Weight function ('uniform', 'distance')
- **algorithm**: Algorithm to use ('auto', 'ball_tree', 'kd_tree', 'brute')
- **leaf_size**: Leaf size for tree algorithms (default: 30)
- **p**: Power parameter for Minkowski metric (default: 2)
- **metric**: Distance metric ('minkowski', 'euclidean', 'manhattan')

## Usage Example
```python
from algorithms.classification.knn.sklearn_impl import KNNSklearn

# Initialize KNN
knn = KNNSklearn(n_neighbors=5, weights='distance', metric='euclidean')

# Train model
knn.fit(X_train, y_train)

# Make predictions
predictions = knn.predict(X_test)
probabilities = knn.predict_proba(X_test)

# Evaluate model
results = knn.evaluate(X_test, y_test)

# Analyze k vs accuracy
knn.plot_k_vs_accuracy(X_train, y_train, X_test, y_test)
```

## References
- Cover, T., & Hart, P. (1967). Nearest neighbor pattern classification
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning
- Scikit-learn Documentation: https://scikit-learn.org/stable/modules/neighbors.html
