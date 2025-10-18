"""
Logistic Regression Implementation using TensorFlow
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class LogisticRegressionTensorFlow:
    """
    Logistic Regression classifier using TensorFlow/Keras implementation.
    
    This class implements logistic regression as a neural network with
    a single dense layer and sigmoid/softmax activation.
    """
    
    def __init__(self, learning_rate=0.01, random_state=42):
        """
        Initialize the Logistic Regression model.
        
        Args:
            learning_rate (float): Learning rate for optimizer
            random_state (int): Random state for reproducibility
        """
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = None
        self.is_fitted = False
        
        # Set random seeds for reproducibility
        tf.random.set_seed(random_state)
        np.random.seed(random_state)
        
    def _build_model(self, input_dim, num_classes):
        """
        Build the neural network model.
        
        Args:
            input_dim (int): Number of input features
            num_classes (int): Number of output classes
        """
        model = Sequential()
        
        if num_classes == 2:
            # Binary classification
            model.add(Dense(1, activation='sigmoid', input_shape=(input_dim,)))
            model.compile(
                optimizer=Adam(learning_rate=self.learning_rate),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
        else:
            # Multi-class classification
            model.add(Dense(num_classes, activation='softmax', input_shape=(input_dim,)))
            model.compile(
                optimizer=Adam(learning_rate=self.learning_rate),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
        return model
        
    def fit(self, X, y, epochs=100, batch_size=32, validation_split=0.2, verbose=1):
        """
        Fit the logistic regression model.
        
        Args:
            X (array-like): Training features
            y (array-like): Training labels
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            validation_split (float): Fraction of data to use for validation
            verbose (int): Verbosity level
        """
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        
        # Determine number of classes
        num_classes = len(np.unique(y))
        input_dim = X_scaled.shape[1]
        
        # Build the model
        self.model = self._build_model(input_dim, num_classes)
        
        # Train the model
        history = self.model.fit(
            X_scaled, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose
        )
        
        self.is_fitted = True
        return history
        
    def predict(self, X):
        """
        Make predictions on new data.
        
        Args:
            X (array-like): Features to predict on
            
        Returns:
            array: Predicted labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        if self.model.output_shape[-1] == 1:
            # Binary classification
            return (predictions > 0.5).astype(int).flatten()
        else:
            # Multi-class classification
            return np.argmax(predictions, axis=1)
            
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Args:
            X (array-like): Features to predict on
            
        Returns:
            array: Class probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
        
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model performance.
        
        Args:
            X_test (array-like): Test features
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
