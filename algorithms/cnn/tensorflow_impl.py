"""
Convolutional Neural Network Implementation using TensorFlow
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class CNNTensorFlow:
    """
    Convolutional Neural Network classifier using TensorFlow/Keras implementation.
    
    This class implements a CNN for image classification with configurable architecture.
    """
    
    def __init__(self, input_shape, num_classes, learning_rate=0.001, random_state=42):
        """
        Initialize the CNN model.
        
        Args:
            input_shape (tuple): Shape of input images (height, width, channels)
            num_classes (int): Number of output classes
            learning_rate (float): Learning rate for optimizer
            random_state (int): Random state for reproducibility
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.random_state = random_state
        
        # Set random seeds for reproducibility
        tf.random.set_seed(random_state)
        np.random.seed(random_state)
        
        self.model = self._build_model()
        self.is_fitted = False
        
    def _build_model(self):
        """
        Build the CNN model architecture.
        
        Returns:
            tf.keras.Model: Compiled CNN model
        """
        model = Sequential([
            # First Convolutional Block
            Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Second Convolutional Block
            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Third Convolutional Block
            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Flatten and Dense Layers
            Flatten(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.5),
            
            # Output Layer
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
        Fit the CNN model.
        
        Args:
            X_train (array-like): Training images
            y_train (array-like): Training labels
            X_val (array-like): Validation images
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
                patience=10,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if validation_data else 'loss',
                factor=0.5,
                patience=5,
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
            X (array-like): Images to predict on
            
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
            X (array-like): Images to predict on
            
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
            X_test (array-like): Test images
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
