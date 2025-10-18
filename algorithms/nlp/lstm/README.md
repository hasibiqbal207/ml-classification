# Long Short-Term Memory (LSTM)

## Overview
LSTM is a type of Recurrent Neural Network (RNN) architecture designed to overcome the vanishing gradient problem in traditional RNNs. It's particularly effective for sequence data and time series analysis.

## LSTM Architecture

### Core Components

#### Forget Gate
```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
```
- Decides what information to discard from cell state
- Uses sigmoid activation (0 = forget, 1 = keep)

#### Input Gate
```
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)
```
- Decides what new information to store
- Creates candidate values for cell state

#### Cell State Update
```
C_t = f_t * C_{t-1} + i_t * C̃_t
```
- Updates the cell state by forgetting old info and adding new info

#### Output Gate
```
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
h_t = o_t * tanh(C_t)
```
- Decides what parts of cell state to output
- Produces the hidden state

## Key Concepts

### Memory Cells
- **Cell State (C_t)**: Long-term memory that flows through the network
- **Hidden State (h_t)**: Short-term memory used for predictions
- **Gates**: Control information flow (forget, input, output)

### Gradient Flow
- **Vanishing Gradient**: Traditional RNNs lose gradient information over time
- **LSTM Solution**: Constant error carousel allows gradients to flow unchanged
- **Backpropagation Through Time**: Gradients flow back through the entire sequence

### Sequence Processing
- **Many-to-One**: Sequence input, single output (classification)
- **Many-to-Many**: Sequence input, sequence output (translation)
- **One-to-Many**: Single input, sequence output (generation)

## Advantages
- **Long-term Dependencies**: Can remember information for long sequences
- **Gradient Stability**: Avoids vanishing gradient problem
- **Flexible Architecture**: Can handle variable-length sequences
- **State-of-the-art Performance**: Excellent for sequential data
- **Bidirectional Processing**: Can process sequences in both directions

## Disadvantages
- **Computational Complexity**: More expensive than simple RNNs
- **Overfitting**: Can memorize training sequences
- **Hyperparameter Sensitivity**: Many parameters to tune
- **Slow Training**: Requires more time than feedforward networks
- **Memory Usage**: Higher memory requirements

## Applications
- **Natural Language Processing**: Sentiment analysis, machine translation
- **Time Series**: Stock prediction, weather forecasting
- **Speech Recognition**: Audio processing and transcription
- **Sequence Classification**: Text classification, activity recognition
- **Anomaly Detection**: Identifying unusual patterns in sequences

## Hyperparameters
- **Hidden Size**: Number of LSTM units
- **Number of Layers**: Depth of the LSTM stack
- **Dropout**: Regularization technique
- **Learning Rate**: Optimization step size
- **Sequence Length**: Length of input sequences
- **Batch Size**: Number of sequences per batch

## Implementation Notes
- Both TensorFlow and PyTorch implementations include:
  - Batch normalization for stable training
  - Dropout for regularization
  - Early stopping to prevent overfitting
  - Learning rate scheduling
- Input sequences should be padded to the same length
- Data should be normalized for better convergence
- Bidirectional LSTMs can improve performance

## References
- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory
- Gers, F. A., et al. (2000). Learning to forget: Continual prediction with LSTM
- Graves, A., et al. (2013). Speech recognition with deep recurrent neural networks
- Sutskever, I., et al. (2014). Sequence to sequence learning with neural networks
