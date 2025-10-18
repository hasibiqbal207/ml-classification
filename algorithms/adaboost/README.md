# AdaBoost Algorithm

## Overview
AdaBoost (Adaptive Boosting) is an ensemble learning method that combines multiple weak learners to create a strong classifier. It adaptively adjusts the weights of training instances and weak learners based on their performance.

## Mathematical Foundation

### Weighted Error
For each weak learner t:

```
ε_t = Σ w_i * I(h_t(x_i) ≠ y_i) / Σ w_i
```

Where:
- `w_i` is the weight of instance i
- `h_t` is the weak learner t
- `I(·)` is an indicator function

### Learner Weight
The weight of weak learner t:

```
α_t = (1/2) * ln((1 - ε_t) / ε_t)
```

### Instance Weight Update
Update weights for next iteration:

```
w_i^(t+1) = w_i^(t) * exp(-α_t * y_i * h_t(x_i))
```

### Final Prediction
Weighted majority vote:

```
H(x) = sign(Σ α_t * h_t(x))
```

## Key Concepts
- **Weak Learners**: Simple classifiers that perform slightly better than random
- **Adaptive Weighting**: Instances are reweighted based on classification difficulty
- **Sequential Learning**: Learners are trained sequentially, not in parallel
- **Margin Theory**: AdaBoost maximizes the margin between classes
- **Exponential Loss**: Minimizes exponential loss function

## Advantages
- Reduces bias and variance
- Works well with weak learners
- Provides feature importance
- Handles both binary and multi-class problems
- Fast training and prediction
- Less prone to overfitting than single models

## Disadvantages
- Sensitive to noisy data and outliers
- Can overfit with too many weak learners
- Requires careful tuning of learning rate
- May not work well with complex weak learners
- Performance can degrade with high-dimensional data

## Implementation Notes
- Uses DecisionTreeClassifier as default weak learner
- Supports both SAMME and SAMME.R algorithms
- Comprehensive analysis of estimator weights and errors
- Learning curve visualization
- Feature importance analysis

## Hyperparameters
- **base_estimator**: Base estimator (default: DecisionTreeClassifier)
- **n_estimators**: Number of estimators (default: 50)
- **learning_rate**: Learning rate (default: 1.0)
- **algorithm**: Algorithm to use ('SAMME', 'SAMME.R')
- **random_state**: Random state for reproducibility

## Usage Example
```python
from algorithms.classification.adaboost.sklearn_impl import AdaBoostSklearn

# Initialize AdaBoost
ada = AdaBoostSklearn(n_estimators=50, learning_rate=1.0)

# Train model
ada.fit(X_train, y_train)

# Make predictions
predictions = ada.predict(X_test)
probabilities = ada.predict_proba(X_test)

# Evaluate model
results = ada.evaluate(X_test, y_test)

# Analyze estimator weights
ada.plot_estimator_weights()
```

## References
- Freund, Y., & Schapire, R. E. (1997). A decision-theoretic generalization of on-line learning
- Schapire, R. E. (1999). A brief introduction to boosting
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning
- Scikit-learn Documentation: https://scikit-learn.org/stable/modules/ensemble.html#adaboost
