#!/usr/bin/env python3
"""
TensorFlow implementation of RNN-based models for multilabel emotion classification.

This module provides LSTM, BiLSTM, and GRU implementations specifically designed
for the GoEmotions dataset with 27 emotion categories.
"""

import os
import pickle
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Embedding, LSTM, Bidirectional, GRU, Dense, Dropout,
    Input, GlobalMaxPooling1D, GlobalAveragePooling1D,
    Concatenate, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    hamming_loss, jaccard_score, classification_report
)
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RNNEmotionClassifier:
    """
    Base class for RNN-based emotion classification models.
    """
    
    def __init__(
        self,
        vocab_size: int = 10000,
        embedding_dim: int = 128,
        rnn_units: int = 64,
        num_emotions: int = 27,
        max_sequence_length: int = 100,
        dropout_rate: float = 0.3,
        learning_rate: float = 0.001,
        model_type: str = "lstm"
    ):
        """
        Initialize RNN emotion classifier.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            rnn_units: Number of RNN units
            num_emotions: Number of emotion categories
            max_sequence_length: Maximum sequence length
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
            model_type: Type of RNN ('lstm', 'bilstm', 'gru')
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.rnn_units = rnn_units
        self.num_emotions = num_emotions
        self.max_sequence_length = max_sequence_length
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model_type = model_type
        
        self.model = None
        self.is_fitted = False
        self.history = None
        self.emotion_categories = None
        
        # Set random seeds for reproducibility
        tf.random.set_seed(42)
        np.random.seed(42)
    
    def _build_model(self) -> Model:
        """
        Build the RNN model architecture.
        
        Returns:
            Compiled Keras model
        """
        # Input layer
        inputs = Input(shape=(self.max_sequence_length,), name='text_input')
        
        # Embedding layer
        embedding = Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim,
            input_length=self.max_sequence_length,
            name='embedding'
        )(inputs)
        
        # RNN layer based on model type
        if self.model_type.lower() == "lstm":
            rnn_layer = LSTM(
                self.rnn_units,
                return_sequences=True,
                dropout=self.dropout_rate,
                recurrent_dropout=self.dropout_rate,
                name='lstm'
            )(embedding)
        elif self.model_type.lower() == "bilstm":
            rnn_layer = Bidirectional(
                LSTM(
                    self.rnn_units,
                    return_sequences=True,
                    dropout=self.dropout_rate,
                    recurrent_dropout=self.dropout_rate,
                    name='bilstm'
                )
            )(embedding)
        elif self.model_type.lower() == "gru":
            rnn_layer = GRU(
                self.rnn_units,
                return_sequences=True,
                dropout=self.dropout_rate,
                recurrent_dropout=self.dropout_rate,
                name='gru'
            )(embedding)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Global pooling
        max_pool = GlobalMaxPooling1D(name='max_pooling')(rnn_layer)
        avg_pool = GlobalAveragePooling1D(name='avg_pooling')(rnn_layer)
        pooled = Concatenate(name='concatenate')([max_pool, avg_pool])
        
        # Dense layers
        dense1 = Dense(128, activation='relu', name='dense1')(pooled)
        dense1 = BatchNormalization(name='batch_norm1')(dense1)
        dense1 = Dropout(self.dropout_rate, name='dropout1')(dense1)
        
        dense2 = Dense(64, activation='relu', name='dense2')(dense1)
        dense2 = BatchNormalization(name='batch_norm2')(dense2)
        dense2 = Dropout(self.dropout_rate, name='dropout2')(dense2)
        
        # Output layer for multilabel classification
        outputs = Dense(
            self.num_emotions,
            activation='sigmoid',
            name='emotion_output'
        )(dense2)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs, name=f'{self.model_type}_emotion_classifier')
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_data: Optional[Tuple] = None,
        epochs: int = 20,
        batch_size: int = 32,
        verbose: int = 1,
        callbacks: Optional[List] = None
    ) -> Dict:
        """
        Train the RNN model.
        
        Args:
            X: Input sequences (samples, max_sequence_length)
            y: Target labels (samples, num_emotions)
            validation_data: Validation data tuple (X_val, y_val)
            epochs: Number of training epochs
            batch_size: Training batch size
            verbose: Verbosity level
            callbacks: List of Keras callbacks
            
        Returns:
            Training history dictionary
        """
        if self.model is None:
            self.model = self._build_model()
        
        # Default callbacks
        if callbacks is None:
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True,
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=3,
                    min_lr=1e-7,
                    verbose=1
                )
            ]
        
        # Train model
        logger.info(f"Training {self.model_type.upper()} model...")
        self.history = self.model.fit(
            X, y,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=callbacks
        )
        
        self.is_fitted = True
        logger.info(f"Training completed. Best validation loss: {min(self.history.history['val_loss']):.4f}")
        
        return self.history.history
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict emotions for input sequences.
        
        Args:
            X: Input sequences
            threshold: Classification threshold
            
        Returns:
            Binary predictions (samples, num_emotions)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict emotion probabilities for input sequences.
        
        Args:
            X: Input sequences
            
        Returns:
            Probability predictions (samples, num_emotions)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        return self.model.predict(X, verbose=0)
    
    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        threshold: float = 0.5,
        emotion_categories: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X: Input sequences
            y: True labels
            threshold: Classification threshold
            emotion_categories: List of emotion category names
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        # Get predictions
        y_pred = self.predict(X, threshold)
        y_proba = self.predict_proba(X)
        
        # Calculate metrics
        metrics = {}
        
        # Accuracy (exact match)
        metrics['accuracy'] = accuracy_score(y, y_pred)
        
        # Multilabel metrics
        metrics['hamming_loss'] = hamming_loss(y, y_pred)
        metrics['jaccard_score'] = jaccard_score(y, y_pred, average='macro', zero_division=0)
        
        # F1 scores
        metrics['f1_macro'] = f1_score(y, y_pred, average='macro', zero_division=0)
        metrics['f1_micro'] = f1_score(y, y_pred, average='micro', zero_division=0)
        metrics['f1_weighted'] = f1_score(y, y_pred, average='weighted', zero_division=0)
        
        # Precision and Recall
        metrics['precision_macro'] = precision_score(y, y_pred, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(y, y_pred, average='macro', zero_division=0)
        metrics['precision_micro'] = precision_score(y, y_pred, average='micro', zero_division=0)
        metrics['recall_micro'] = recall_score(y, y_pred, average='micro', zero_division=0)
        
        return metrics
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """
        Plot training history.
        
        Args:
            save_path: Path to save the plot
        """
        if self.history is None:
            raise ValueError("No training history available")
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        axes[0].plot(self.history.history['loss'], label='Training Loss')
        if 'val_loss' in self.history.history:
            axes[0].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0].set_title(f'{self.model_type.upper()} Model Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot accuracy
        axes[1].plot(self.history.history['accuracy'], label='Training Accuracy')
        if 'val_accuracy' in self.history.history:
            axes[1].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[1].set_title(f'{self.model_type.upper()} Model Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def save_model(self, filepath: str):
        """
        Save the trained model.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load a trained model.
        
        Args:
            filepath: Path to the saved model
        """
        self.model = tf.keras.models.load_model(filepath)
        self.is_fitted = True
        logger.info(f"Model loaded from {filepath}")


class LSTMEmotionClassifier(RNNEmotionClassifier):
    """LSTM-based emotion classifier."""
    
    def __init__(self, **kwargs):
        kwargs['model_type'] = 'lstm'
        super().__init__(**kwargs)


class BiLSTMEmotionClassifier(RNNEmotionClassifier):
    """Bidirectional LSTM-based emotion classifier."""
    
    def __init__(self, **kwargs):
        kwargs['model_type'] = 'bilstm'
        super().__init__(**kwargs)


class GRUEmotionClassifier(RNNEmotionClassifier):
    """GRU-based emotion classifier."""
    
    def __init__(self, **kwargs):
        kwargs['model_type'] = 'gru'
        super().__init__(**kwargs)


def create_text_sequences(
    texts: List[str],
    tokenizer: Any,
    max_length: int = 100
) -> np.ndarray:
    """
    Convert texts to sequences for RNN input.
    
    Args:
        texts: List of text strings
        tokenizer: Fitted tokenizer
        max_length: Maximum sequence length
        
    Returns:
        Padded sequences array
    """
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(
        sequences, maxlen=max_length, padding='post', truncating='post'
    )
    return padded_sequences


def create_tokenizer(
    texts: List[str],
    vocab_size: int = 10000,
    oov_token: str = "<OOV>"
) -> Any:
    """
    Create and fit a tokenizer for text sequences.
    
    Args:
        texts: List of text strings for fitting
        vocab_size: Maximum vocabulary size
        oov_token: Token for out-of-vocabulary words
        
    Returns:
        Fitted tokenizer
    """
    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        num_words=vocab_size,
        oov_token=oov_token,
        filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    )
    tokenizer.fit_on_texts(texts)
    return tokenizer


def save_tokenizer(tokenizer: Any, filepath: str):
    """
    Save tokenizer to file.
    
    Args:
        tokenizer: Fitted tokenizer
        filepath: Path to save the tokenizer
    """
    with open(filepath, 'wb') as f:
        pickle.dump(tokenizer, f)
    logger.info(f"Tokenizer saved to {filepath}")


def load_tokenizer(filepath: str) -> Any:
    """
    Load tokenizer from file.
    
    Args:
        filepath: Path to the saved tokenizer
        
    Returns:
        Loaded tokenizer
    """
    with open(filepath, 'rb') as f:
        tokenizer = pickle.load(f)
    logger.info(f"Tokenizer loaded from {filepath}")
    return tokenizer


# Example usage and testing
if __name__ == "__main__":
    # This would be used for testing the implementation
    print("RNN Emotion Classification Models")
    print("Available models: LSTM, BiLSTM, GRU")
    print("Designed for GoEmotions multilabel classification")
