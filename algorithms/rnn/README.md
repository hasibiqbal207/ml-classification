# Recurrent Neural Networks (RNN) for Emotion Classification

This directory contains RNN-based implementations for multilabel emotion classification, specifically designed for the GoEmotions dataset.

## 🧠 Models Available

### 1. **LSTM (Long Short-Term Memory)**
- **File**: `tensorflow_impl.py`
- **Architecture**: LSTM layers + Dense layers + Logistic Regression
- **Use Case**: Sequential text understanding with memory
- **Best For**: Long texts, complex emotional contexts

### 2. **BiLSTM (Bidirectional LSTM)**
- **File**: `tensorflow_impl.py`
- **Architecture**: Bidirectional LSTM + Dense layers + Logistic Regression
- **Use Case**: Context from both past and future
- **Best For**: Complex emotional expressions, nuanced sentiment

### 3. **GRU (Gated Recurrent Unit)**
- **File**: `tensorflow_impl.py`
- **Architecture**: GRU layers + Dense layers + Logistic Regression
- **Use Case**: Lighter alternative to LSTM
- **Best For**: Faster training, similar performance to LSTM

## 🎯 Key Features

### **Multilabel Classification**
- **Output**: 27 emotion categories (GoEmotions dataset)
- **Activation**: Sigmoid for multilabel probabilities
- **Loss Function**: Binary crossentropy for multilabel
- **Metrics**: Hamming Loss, Jaccard Score, F1-macro, F1-micro

### **Text Processing**
- **Word Embeddings**: Pre-trained or trainable embeddings
- **Sequence Padding**: Variable length text handling
- **Vocabulary**: Configurable vocabulary size
- **Preprocessing**: Text cleaning and tokenization

### **Architecture Components**
- **Embedding Layer**: Word-to-vector conversion
- **RNN Layers**: LSTM/BiLSTM/GRU for sequence processing
- **Dense Layers**: Feature extraction and dimensionality reduction
- **Output Layer**: Multilabel classification with sigmoid activation

## 🚀 Usage

### **Training**
```python
from algorithms.rnn.tensorflow_impl import LSTMEmotionClassifier

# Initialize model
model = LSTMEmotionClassifier(
    vocab_size=10000,
    embedding_dim=128,
    lstm_units=64,
    num_emotions=27,
    max_sequence_length=100
)

# Train model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10)
```

### **Prediction**
```python
# Predict emotions
predictions = model.predict(X_test)

# Get probabilities
probabilities = model.predict_proba(X_test)
```

## 📊 Expected Performance

### **GoEmotions Dataset**
- **F1-macro**: 0.45-0.60 (vs 0.35-0.45 for traditional ML)
- **Hamming Loss**: 0.10-0.20 (vs 0.15-0.25 for traditional ML)
- **Jaccard Score**: 0.30-0.45 (vs 0.20-0.35 for traditional ML)

### **Training Time**
- **LSTM**: 2-5 minutes per epoch (GoEmotions dataset)
- **BiLSTM**: 3-6 minutes per epoch
- **GRU**: 1.5-4 minutes per epoch

## 🔧 Configuration

### **Model Parameters**
- `vocab_size`: Vocabulary size (default: 10000)
- `embedding_dim`: Word embedding dimension (default: 128)
- `lstm_units`: LSTM hidden units (default: 64)
- `num_emotions`: Number of emotion categories (default: 27)
- `max_sequence_length`: Maximum text length (default: 100)

### **Training Parameters**
- `epochs`: Number of training epochs (default: 20)
- `batch_size`: Training batch size (default: 32)
- `learning_rate`: Optimizer learning rate (default: 0.001)
- `dropout_rate`: Dropout for regularization (default: 0.3)

## 🎯 Advantages Over Traditional ML

### **1. Sequential Understanding**
- Captures word order and context
- Better understanding of emotional expressions
- Handles complex sentence structures

### **2. Multilabel Performance**
- Learns relationships between emotions
- Better handling of multiple simultaneous emotions
- Improved accuracy on complex emotional texts

### **3. Text Representation**
- Dense vector representations
- Better handling of synonyms and context
- Reduced vocabulary issues

## 🔬 Research Background

### **Why RNNs for Emotion Classification?**
1. **Sequential Nature**: Emotions in text depend on word order and context
2. **Long-range Dependencies**: Emotional context can span across sentences
3. **Multilabel Complexity**: RNNs better handle multiple simultaneous emotions
4. **Contextual Understanding**: Better capture of nuanced emotional expressions

### **LSTM vs BiLSTM vs GRU**
- **LSTM**: Good balance of performance and complexity
- **BiLSTM**: Best performance, captures bidirectional context
- **GRU**: Faster training, similar performance to LSTM

## 📚 References

- [GoEmotions Dataset](https://github.com/google-research/google-research/tree/master/goemotions)
- [LSTM for Text Classification](https://arxiv.org/abs/1506.00019)
- [Bidirectional LSTM for Emotion Recognition](https://ieeexplore.ieee.org/document/8255365)
- [Multilabel Text Classification with RNNs](https://arxiv.org/abs/1609.06647)
