"""
Long Short-Term Memory (LSTM) Implementation using TensorFlow
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class LSTMTensorFlow:
    """
    LSTM classifier using TensorFlow/Keras implementation.
    
    This class implements an LSTM network for sequence classification.
    """
    
    def __init__(self, sequence_length, num_features, num_classes, 
                 lstm_units=128, learning_rate=0.001, random_state=42):
        """
        Initialize the LSTM model.
        
        Args:
            sequence_length (int): Length of input sequences
            num_features (int): Number of features per time step
            num_classes (int): Number of output classes
            lstm_units (int): Number of LSTM units
            learning_rate (float): Learning rate for optimizer
            random_state (int): Random state for reproducibility
        """
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.num_classes = num_classes
        self.lstm_units = lstm_units
        self.learning_rate = learning_rate
        self.random_state = random_state
        
        # Set random seeds for reproducibility
        tf.random.set_seed(random_state)
        np.random.seed(random_state)
        
        self.model = self._build_model()
        self.is_fitted = False
        
    def _build_model(self):
        """
        Build the LSTM model architecture.
        
        Returns:
            tf.keras.Model: Compiled LSTM model
        """
        model = Sequential([
            # First LSTM layer
            LSTM(self.lstm_units, return_sequences=True, 
                 input_shape=(self.sequence_length, self.num_features)),
            BatchNormalization(),
            Dropout(0.2),
            
            # Second LSTM layer
            LSTM(self.lstm_units // 2, return_sequences=False),
            BatchNormalization(),
            Dropout(0.2),
            
            # Dense layers
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.3),
            
            # Output layer
            Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
        
    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=100, 
            batch_size=32, verbose=1):
        """
        Fit the LSTM model.
        
        Args:
            X_train (array-like): Training sequences
            y_train (array-like): Training labels
            X_val (array-like): Validation sequences
            y_val (array-like): Validation labels
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            verbose (int): Verbosity level
            
        Returns:
            tf.keras.callbacks.History: Training history
        """
        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
            
        # Define callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if validation_data else 'loss',
                patience=15,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if validation_data else 'loss',
                factor=0.5,
                patience=7,
                min_lr=1e-7
            )
        ]
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        self.is_fitted = True
        return history
        
    def predict(self, X):
        """
        Make predictions on new data.
        
        Args:
            X (array-like): Sequences to predict on
            
        Returns:
            array: Predicted labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        predictions = self.model.predict(X)
        return np.argmax(predictions, axis=1)
        
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Args:
            X (array-like): Sequences to predict on
            
        Returns:
            array: Class probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        return self.model.predict(X)
        
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model performance.
        
        Args:
            X_test (array-like): Test sequences
            y_test (array-like): Test labels
            
        Returns:
            dict: Evaluation metrics
        """
        y_pred = self.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        return {
            'accuracy': accuracy,
            'predictions': y_pred,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
    def get_model_summary(self):
        """
        Get model architecture summary.
        
        Returns:
            str: Model summary
        """
        return self.model.summary()
