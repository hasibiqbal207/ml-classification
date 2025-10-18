# Random Forest Algorithm

## Overview
Random Forest is an ensemble learning method that combines multiple decision trees to create a more robust and accurate classifier. It uses bagging (bootstrap aggregating) to reduce overfitting and improve generalization.

## Mathematical Foundation

### Bootstrap Aggregating (Bagging)
Random Forest uses bootstrap sampling to create multiple training sets:

```
D_i = Bootstrap(D) for i = 1 to B
```

Where:
- `D` is the original training set
- `B` is the number of trees
- `D_i` is the i-th bootstrap sample

### Random Feature Selection
At each split, only a random subset of features is considered:

```
m = sqrt(p) for classification
m = p/3 for regression
```

Where:
- `p` is the total number of features
- `m` is the number of features to consider at each split

### Final Prediction
For classification, the final prediction is the majority vote:

```
ŷ = mode(T_1(x), T_2(x), ..., T_B(x))
```

## Key Concepts
- **Bootstrap Sampling**: Random sampling with replacement
- **Feature Randomness**: Random subset of features at each split
- **Voting**: Majority vote for classification, average for regression
- **Out-of-Bag (OOB) Error**: Unbiased estimate of generalization error
- **Feature Importance**: Measure of feature contribution to predictions

## Advantages
- Reduces overfitting compared to single decision trees
- Handles missing values well
- Provides feature importance scores
- Works well with both numerical and categorical data
- Robust to outliers
- Can handle large datasets efficiently

## Disadvantages
- Less interpretable than single decision trees
- Can be slow for very large datasets
- May overfit on noisy data
- Memory intensive for large numbers of trees
- Biased towards features with many categories

## Implementation Notes
- Automatic feature scaling is not required
- Supports parallel processing
- Includes comprehensive evaluation metrics
- Feature importance analysis capabilities
- Tree depth and structure analysis

## Hyperparameters
- **n_estimators**: Number of trees in the forest (default: 100)
- **max_depth**: Maximum depth of trees (default: None)
- **min_samples_split**: Minimum samples to split a node (default: 2)
- **min_samples_leaf**: Minimum samples at a leaf node (default: 1)
- **random_state**: Random state for reproducibility
- **n_jobs**: Number of parallel jobs (default: -1)

## Usage Example
```python
from algorithms.classification.random_forest.sklearn_impl import RandomForestSklearn

# Initialize Random Forest
rf = RandomForestSklearn(n_estimators=100, max_depth=10, random_state=42)

# Train model
rf.fit(X_train, y_train)

# Make predictions
predictions = rf.predict(X_test)
probabilities = rf.predict_proba(X_test)

# Evaluate model
results = rf.evaluate(X_test, y_test)

# Analyze feature importance
rf.plot_feature_importance(feature_names)
```

## References
- Breiman, L. (2001). Random forests
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning
- Scikit-learn Documentation: https://scikit-learn.org/stable/modules/ensemble.html#random-forests
