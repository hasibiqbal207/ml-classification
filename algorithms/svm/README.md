# Support Vector Machine (SVM) Algorithm

## Overview
Support Vector Machine is a powerful supervised learning algorithm used for classification and regression tasks. SVM finds the optimal hyperplane that separates classes with maximum margin.

## Mathematical Foundation

### Linear SVM
For linearly separable data, SVM finds the hyperplane that maximizes the margin:

```
w^T * x + b = 0
```

The optimization problem:
```
minimize: (1/2) * ||w||^2 + C * Σ(ξ_i)
subject to: y_i(w^T * x_i + b) ≥ 1 - ξ_i
```

Where:
- `w` is the weight vector
- `b` is the bias term
- `C` is the regularization parameter
- `ξ_i` are slack variables for non-separable data

### Kernel Trick
For non-linearly separable data, SVM uses kernel functions to map data to higher dimensions:

- **Linear**: `K(x_i, x_j) = x_i^T * x_j`
- **Polynomial**: `K(x_i, x_j) = (γ * x_i^T * x_j + r)^d`
- **RBF**: `K(x_i, x_j) = exp(-γ * ||x_i - x_j||^2)`
- **Sigmoid**: `K(x_i, x_j) = tanh(γ * x_i^T * x_j + r)`

## Key Concepts
- **Support Vectors**: Data points closest to the decision boundary
- **Margin**: Distance between the decision boundary and support vectors
- **Kernel Trick**: Method to handle non-linear data by mapping to higher dimensions
- **Regularization**: C parameter controls the trade-off between margin maximization and error minimization

## Advantages
- Effective in high-dimensional spaces
- Memory efficient (uses only support vectors)
- Versatile (different kernel functions)
- Works well with small to medium datasets
- Less prone to overfitting with proper regularization

## Disadvantages
- Poor performance on large datasets
- Sensitive to feature scaling
- No probabilistic output (unless probability=True)
- Black box model (difficult to interpret)
- Sensitive to noise and outliers

## Implementation Notes
- Automatic feature scaling is included
- Supports probability estimates
- Includes visualization methods for 2D data
- Comprehensive evaluation metrics
- Support vector analysis capabilities

## Hyperparameters
- **C**: Regularization parameter (default: 1.0)
- **kernel**: Kernel type ('linear', 'poly', 'rbf', 'sigmoid')
- **gamma**: Kernel coefficient (default: 'scale')
- **random_state**: Random state for reproducibility

## Usage Example
```python
from algorithms.classification.svm.sklearn_impl import SVMSklearn

# Initialize SVM
svm = SVMSklearn(C=1.0, kernel='rbf', gamma='scale')

# Train model
svm.fit(X_train, y_train)

# Make predictions
predictions = svm.predict(X_test)
probabilities = svm.predict_proba(X_test)

# Evaluate model
results = svm.evaluate(X_test, y_test)
```

## References
- Cortes, C., & Vapnik, V. (1995). Support-vector networks
- Vapnik, V. (1998). Statistical Learning Theory
- Scikit-learn Documentation: https://scikit-learn.org/stable/modules/svm.html
