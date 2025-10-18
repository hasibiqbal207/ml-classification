# Multi-Layer Perceptron (MLP) Algorithm

## Overview
Multi-Layer Perceptron is a feedforward artificial neural network that consists of multiple layers of perceptrons with nonlinear activation functions. It's a powerful deep learning model for complex, high-dimensional data.

## Mathematical Foundation

### Forward Propagation
For a network with L layers:

```
a⁽⁰⁾ = x (input)
z⁽ˡ⁾ = W⁽ˡ⁾a⁽ˡ⁻¹⁾ + b⁽ˡ⁾
a⁽ˡ⁾ = g(z⁽ˡ⁾)
```

Where:
- `W⁽ˡ⁾` is the weight matrix for layer l
- `b⁽ˡ⁾` is the bias vector for layer l
- `g(·)` is the activation function
- `a⁽ˡ⁾` is the activation of layer l

### Activation Functions

**ReLU:**
```
g(z) = max(0, z)
```

**Sigmoid:**
```
g(z) = 1 / (1 + e^(-z))
```

**Tanh:**
```
g(z) = (e^z - e^(-z)) / (e^z + e^(-z))
```

### Loss Functions

**Binary Classification:**
```
L = -[y*log(ŷ) + (1-y)*log(1-ŷ)]
```

**Multi-class Classification:**
```
L = -Σ y_i * log(ŷ_i)
```

### Backpropagation
Gradient descent with chain rule:

```
∂L/∂W⁽ˡ⁾ = ∂L/∂z⁽ˡ⁾ * (a⁽ˡ⁻¹⁾)ᵀ
∂L/∂b⁽ˡ⁾ = ∂L/∂z⁽ˡ⁾
```

## Key Concepts
- **Feedforward**: Information flows from input to output
- **Backpropagation**: Gradient computation using chain rule
- **Activation Functions**: Non-linear transformations
- **Regularization**: Dropout, batch normalization, weight decay
- **Optimization**: Adam, SGD, RMSprop optimizers

## Advantages
- Can approximate any continuous function
- Handles non-linear relationships well
- Works with high-dimensional data
- Flexible architecture
- Can learn complex patterns

## Disadvantages
- Requires large amounts of data
- Computationally expensive
- Prone to overfitting
- Black box model
- Sensitive to hyperparameters
- Requires feature scaling

## Implementation Notes
- Automatic feature scaling included
- Supports both binary and multi-class classification
- Includes regularization techniques (dropout, batch normalization)
- Comprehensive training history visualization
- Model architecture analysis capabilities

## Hyperparameters
- **hidden_layers**: Number of neurons in each hidden layer (default: [128, 64])
- **activation**: Activation function ('relu', 'sigmoid', 'tanh')
- **dropout_rate**: Dropout rate for regularization (default: 0.3)
- **learning_rate**: Learning rate for optimizer (default: 0.001)
- **batch_size**: Batch size for training (default: 32)
- **epochs**: Number of training epochs (default: 100)

## Usage Example
```python
from algorithms.classification.mlp.tensorflow_impl import MLPTensorFlow

# Initialize MLP
mlp = MLPTensorFlow(hidden_layers=[128, 64], activation='relu', dropout_rate=0.3)

# Train model
mlp.fit(X_train, y_train)

# Make predictions
predictions = mlp.predict(X_test)
probabilities = mlp.predict_proba(X_test)

# Evaluate model
results = mlp.evaluate(X_test, y_test)

# Plot training history
mlp.plot_training_history()
```

## References
- Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning
- TensorFlow Documentation: https://www.tensorflow.org/api_docs/python/tf/keras
