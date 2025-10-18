# Naive Bayes Algorithm

## Overview
Naive Bayes is a probabilistic classification algorithm based on Bayes' theorem with the "naive" assumption of conditional independence between features. Despite this simplifying assumption, it often performs surprisingly well in practice.

## Mathematical Foundation

### Bayes' Theorem
```
P(y|x) = P(x|y) * P(y) / P(x)
```

Where:
- `P(y|x)` is the posterior probability
- `P(x|y)` is the likelihood
- `P(y)` is the prior probability
- `P(x)` is the evidence

### Naive Assumption
Features are conditionally independent given the class:

```
P(x|y) = P(x₁|y) * P(x₂|y) * ... * P(xₙ|y)
```

### Classification Rule
Assign the class with highest posterior probability:

```
ŷ = argmax_y P(y|x) = argmax_y P(y) * ∏P(xᵢ|y)
```

## Variants

### Gaussian Naive Bayes
For continuous features, assumes Gaussian distribution:

```
P(xᵢ|y) = (1/√(2πσ²)) * exp(-(xᵢ-μ)²/(2σ²))
```

### Multinomial Naive Bayes
For discrete features (e.g., word counts):

```
P(xᵢ|y) = (Nᵢ + α) / (N + α * n)
```

### Bernoulli Naive Bayes
For binary features:

```
P(xᵢ|y) = pᵢ^xᵢ * (1-pᵢ)^(1-xᵢ)
```

## Key Concepts
- **Prior Probability**: Probability of each class before seeing data
- **Likelihood**: Probability of features given the class
- **Posterior Probability**: Probability of class given features
- **Smoothing**: Additive smoothing to handle zero probabilities
- **Log Probabilities**: Use log to avoid numerical underflow

## Advantages
- Simple and fast
- Works well with small datasets
- Handles multiple classes naturally
- Not sensitive to irrelevant features
- Provides probability estimates
- Works well with text classification

## Disadvantages
- Strong independence assumption
- Can be outperformed by more sophisticated methods
- Sensitive to feature scaling (Gaussian variant)
- May not work well with correlated features
- Limited expressiveness due to independence assumption

## Implementation Notes
- Automatic feature scaling for Gaussian variant
- MinMax scaling for Multinomial and Bernoulli variants
- Supports probability and log-probability predictions
- Comprehensive visualization methods
- Class prior analysis capabilities

## Hyperparameters
- **variant**: Type of Naive Bayes ('gaussian', 'multinomial', 'bernoulli')
- **alpha**: Smoothing parameter (default: 1.0)
- **fit_prior**: Whether to learn class priors (default: True)
- **class_prior**: Prior probabilities of classes (default: None)
- **binarize**: Threshold for binarizing features (default: 0.0)

## Usage Example
```python
from algorithms.classification.naive_bayes.sklearn_impl import NaiveBayesSklearn

# Initialize Naive Bayes
nb = NaiveBayesSklearn(variant='gaussian', alpha=1.0)

# Train model
nb.fit(X_train, y_train)

# Make predictions
predictions = nb.predict(X_test)
probabilities = nb.predict_proba(X_test)

# Evaluate model
results = nb.evaluate(X_test, y_test)

# Analyze class priors
nb.plot_class_priors()
```

## References
- McCallum, A., & Nigam, K. (1998). A comparison of event models for naive bayes text classification
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning
- Scikit-learn Documentation: https://scikit-learn.org/stable/modules/naive_bayes.html
