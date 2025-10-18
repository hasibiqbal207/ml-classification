"""
TensorFlow training utilities.
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy, SparseCategoricalCrossentropy
import os
from pathlib import Path


class TensorFlowTrainer:
    """
    Utility class for training TensorFlow models.
    """
    
    def __init__(self, model, model_name="model"):
        """
        Initialize TensorFlowTrainer.
        
        Args:
            model: TensorFlow/Keras model
            model_name (str): Name of the model for saving
        """
        self.model = model
        self.model_name = model_name
        self.history = None
        
    def compile_model(self, optimizer='adam', loss='sparse_categorical_crossentropy', 
                     metrics=['accuracy'], learning_rate=0.001):
        """
        Compile the model with specified parameters.
        
        Args:
            optimizer (str): Optimizer name or instance
            loss (str): Loss function name or instance
            metrics (list): List of metrics
            learning_rate (float): Learning rate for optimizer
        """
        # Set up optimizer
        if optimizer == 'adam':
            opt = Adam(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            opt = SGD(learning_rate=learning_rate)
        else:
            opt = optimizer
            
        # Set up loss function
        if loss == 'binary_crossentropy':
            loss_fn = BinaryCrossentropy()
        elif loss == 'categorical_crossentropy':
            loss_fn = CategoricalCrossentropy()
        elif loss == 'sparse_categorical_crossentropy':
            loss_fn = SparseCategoricalCrossentropy()
        else:
            loss_fn = loss
            
        self.model.compile(optimizer=opt, loss=loss_fn, metrics=metrics)
        
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              epochs=100, batch_size=32, verbose=1, callbacks=None):
        """
        Train the model.
        
        Args:
            X_train (array-like): Training features
            y_train (array-like): Training labels
            X_val (array-like): Validation features
            y_val (array-like): Validation labels
            epochs (int): Number of epochs
            batch_size (int): Batch size
            verbose (int): Verbosity level
            callbacks (list): List of callbacks
            
        Returns:
            tf.keras.callbacks.History: Training history
        """
        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
            
        # Set up default callbacks if none provided
        if callbacks is None:
            callbacks = self._get_default_callbacks()
            
        # Train the model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return self.history
        
    def _get_default_callbacks(self):
        """
        Get default callbacks for training.
        
        Returns:
            list: List of callbacks
        """
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Add model checkpoint if model name is provided
        if self.model_name:
            checkpoint_path = f"models/{self.model_name}_best.h5"
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            
            callbacks.append(
                ModelCheckpoint(
                    filepath=checkpoint_path,
                    monitor='val_loss',
                    save_best_only=True,
                    verbose=1
                )
            )
            
        return callbacks
        
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data.
        
        Args:
            X_test (array-like): Test features
            y_test (array-like): Test labels
            
        Returns:
            dict: Evaluation results
        """
        results = self.model.evaluate(X_test, y_test, verbose=0)
        
        # Create results dictionary
        metrics_names = self.model.metrics_names
        results_dict = dict(zip(metrics_names, results))
        
        return results_dict
        
    def predict(self, X):
        """
        Make predictions on new data.
        
        Args:
            X (array-like): Input data
            
        Returns:
            array: Predictions
        """
        return self.model.predict(X)
        
    def save_model(self, filepath=None):
        """
        Save the trained model.
        
        Args:
            filepath (str): Path to save the model
        """
        if filepath is None:
            filepath = f"models/{self.model_name}_final.h5"
            
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
        
    def load_model(self, filepath):
        """
        Load a saved model.
        
        Args:
            filepath (str): Path to the saved model
        """
        self.model = tf.keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
        
    def plot_training_history(self, figsize=(15, 5)):
        """
        Plot training history.
        
        Args:
            figsize (tuple): Figure size
        """
        if self.history is None:
            print("No training history available")
            return
            
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Plot loss
        axes[0].plot(self.history.history['loss'], label='Training Loss')
        if 'val_loss' in self.history.history:
            axes[0].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0].set_title('Model Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot accuracy
        axes[1].plot(self.history.history['accuracy'], label='Training Accuracy')
        if 'val_accuracy' in self.history.history:
            axes[1].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[1].set_title('Model Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.show()


def create_tensorflow_model(input_shape, num_classes, model_type='dense'):
    """
    Create a TensorFlow model based on the specified type.
    
    Args:
        input_shape (tuple): Input shape
        num_classes (int): Number of output classes
        model_type (str): Type of model ('dense', 'cnn', 'lstm')
        
    Returns:
        tf.keras.Model: Created model
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, LSTM, Dropout
    
    model = Sequential()
    
    if model_type == 'dense':
        model.add(Dense(128, activation='relu', input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(num_classes, activation='softmax'))
        
    elif model_type == 'cnn':
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(num_classes, activation='softmax'))
        
    elif model_type == 'lstm':
        model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(64, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
        
    else:
        raise ValueError("Model type must be 'dense', 'cnn', or 'lstm'")
        
    return model
