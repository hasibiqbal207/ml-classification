# Logistic Regression Algorithm

## Overview
Logistic Regression is a linear classification algorithm that uses the logistic function to model the probability of a binary or multi-class outcome.

## Mathematical Foundation

### Binary Classification
For binary classification, the logistic function is:
```
P(y=1|x) = 1 / (1 + e^(-z))
```
where z = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ

### Multi-class Classification
For multi-class classification, we use the softmax function:
```
P(y=k|x) = e^(z_k) / Σ(e^(z_j)) for j=1 to K
```

## Key Concepts
- **Sigmoid Function**: Maps any real number to a value between 0 and 1
- **Decision Boundary**: Linear decision boundary in feature space
- **Maximum Likelihood Estimation**: Used to estimate model parameters
- **Regularization**: L1 (Lasso) and L2 (Ridge) regularization can be applied

## Advantages
- Simple and interpretable
- Fast training and prediction
- Works well with linearly separable data
- Provides probability estimates
- Less prone to overfitting with regularization

## Disadvantages
- Assumes linear relationship between features and log-odds
- Sensitive to outliers
- May not perform well with non-linear relationships
- Requires feature scaling for optimal performance

## Implementation Notes
- All implementations include automatic feature scaling
- TensorFlow and PyTorch versions support both binary and multi-class classification
- Scikit-learn version provides the most comprehensive set of hyperparameters
- All versions include evaluation metrics and confusion matrix

## References
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning
- Bishop, C. M. (2006). Pattern Recognition and Machine Learning
- Scikit-learn Documentation: https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
