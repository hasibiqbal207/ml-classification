# Decision Tree Algorithm

## Overview
Decision Trees are non-parametric supervised learning algorithms that create a model predicting the value of a target variable by learning simple decision rules inferred from data features.

## How Decision Trees Work

### Tree Structure
- **Root Node**: Contains all data
- **Internal Nodes**: Split data based on feature conditions
- **Leaf Nodes**: Final predictions/classifications

### Splitting Criteria

#### Gini Impurity
```
Gini = 1 - Σ(p_i)²
```
where p_i is the probability of class i

#### Entropy
```
Entropy = -Σ(p_i * log₂(p_i))
```

### Information Gain
```
IG = Entropy(parent) - Σ(n_i/n * Entropy(child_i))
```

## Key Concepts
- **Recursive Partitioning**: Tree is built by recursively splitting nodes
- **Feature Selection**: Best feature for splitting is chosen based on impurity measures
- **Pruning**: Removing branches that don't improve generalization
- **Overfitting**: Trees can easily overfit to training data

## Advantages
- Easy to understand and interpret
- Requires little data preparation
- Handles both numerical and categorical data
- Can model non-linear relationships
- Provides feature importance scores
- Robust to outliers

## Disadvantages
- Prone to overfitting
- Unstable (small changes in data can lead to different trees)
- Biased towards features with more levels
- Can create biased trees if some classes dominate
- Difficult to capture linear relationships

## Hyperparameters
- **max_depth**: Maximum depth of the tree
- **min_samples_split**: Minimum samples required to split a node
- **min_samples_leaf**: Minimum samples required at a leaf node
- **criterion**: Function to measure split quality ('gini' or 'entropy')
- **max_features**: Number of features to consider for best split

## Implementation Notes
- Scikit-learn version provides full decision tree functionality with visualization
- PyTorch version is a neural network approximation for comparison
- Both implementations include evaluation metrics and feature importance

## References
- Breiman, L., Friedman, J., Stone, C. J., & Olshen, R. A. (1984). Classification and regression trees
- Quinlan, J. R. (1986). Induction of decision trees
- Scikit-learn Documentation: https://scikit-learn.org/stable/modules/tree.html
