"""
Multi-Layer Perceptron (MLP) Algorithm Implementation

MLP is a feedforward artificial neural network that consists of multiple layers
of perceptrons with nonlinear activation functions.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class MLPTensorFlow:
    """
    Multi-Layer Perceptron implementation using TensorFlow/Keras.
    """
    
    def __init__(self, hidden_layers=[128, 64], activation='relu', dropout_rate=0.3,
                 learning_rate=0.001, batch_size=32, epochs=100, random_state=42):
        """
        Initialize MLP model.
        
        Args:
            hidden_layers (list): Number of neurons in each hidden layer
            activation (str): Activation function ('relu', 'sigmoid', 'tanh')
            dropout_rate (float): Dropout rate for regularization
            learning_rate (float): Learning rate for optimizer
            batch_size (int): Batch size for training
            epochs (int): Number of training epochs
            random_state (int): Random state for reproducibility
        """
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.random_state = random_state
        
        # Set random seeds
        tf.random.set_seed(random_state)
        np.random.seed(random_state)
        
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
        self.history = None
        
    def _build_model(self, input_dim, num_classes):
        """
        Build the MLP model architecture.
        
        Args:
            input_dim (int): Number of input features
            num_classes (int): Number of output classes
        """
        self.model = Sequential()
        
        # Input layer
        self.model.add(Dense(self.hidden_layers[0], activation=self.activation, 
                           input_shape=(input_dim,)))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(self.dropout_rate))
        
        # Hidden layers
        for neurons in self.hidden_layers[1:]:
            self.model.add(Dense(neurons, activation=self.activation))
            self.model.add(BatchNormalization())
            self.model.add(Dropout(self.dropout_rate))
        
        # Output layer
        if num_classes == 2:
            self.model.add(Dense(1, activation='sigmoid'))
            self.loss = 'binary_crossentropy'
            self.metrics = ['accuracy']
        else:
            self.model.add(Dense(num_classes, activation='softmax'))
            self.loss = 'categorical_crossentropy'
            self.metrics = ['accuracy']
            
        # Compile model
        optimizer = Adam(learning_rate=self.learning_rate)
        self.model.compile(optimizer=optimizer, loss=self.loss, metrics=self.metrics)
        
    def fit(self, X, y, validation_split=0.2, verbose=1):
        """
        Train the MLP model.
        
        Args:
            X (array-like): Training features
            y (array-like): Training labels
            validation_split (float): Fraction of data to use for validation
            verbose (int): Verbosity level
            
        Returns:
            self: Fitted model
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Get model dimensions
        input_dim = X_scaled.shape[1]
        num_classes = len(np.unique(y_encoded))
        
        # Build model
        self._build_model(input_dim, num_classes)
        
        # Prepare labels for training
        if num_classes == 2:
            y_train = y_encoded.astype(np.float32)
        else:
            y_train = tf.keras.utils.to_categorical(y_encoded, num_classes)
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
        ]
        
        # Train model
        self.history = self.model.fit(
            X_scaled, y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose
        )
        
        self.is_fitted = True
        return self
        
    def predict(self, X):
        """
        Make predictions.
        
        Args:
            X (array-like): Features to predict
            
        Returns:
            array: Predicted labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        if len(self.label_encoder.classes_) == 2:
            # Binary classification
            predicted_labels = (predictions > 0.5).astype(int).flatten()
        else:
            # Multi-class classification
            predicted_labels = np.argmax(predictions, axis=1)
            
        return self.label_encoder.inverse_transform(predicted_labels)
        
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Args:
            X (array-like): Features to predict
            
        Returns:
            array: Class probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        X_scaled = self.scaler.transform(X)
        probabilities = self.model.predict(X_scaled)
        
        if len(self.label_encoder.classes_) == 2:
            # Binary classification - return probabilities for both classes
            prob_class_1 = probabilities.flatten()
            prob_class_0 = 1 - prob_class_1
            return np.column_stack([prob_class_0, prob_class_1])
        else:
            # Multi-class classification
            return probabilities
            
    def score(self, X, y):
        """
        Calculate accuracy score.
        
        Args:
            X (array-like): Features
            y (array-like): True labels
            
        Returns:
            float: Accuracy score
        """
        predictions = self.predict(X)
        return accuracy_score(y, predictions)
        
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model.
        
        Args:
            X_test (array-like): Test features
            y_test (array-like): Test labels
            
        Returns:
            dict: Evaluation metrics
        """
        predictions = self.predict(X_test)
        probabilities = self.predict_proba(X_test)
        
        accuracy = accuracy_score(y_test, predictions)
        
        # Get classification report
        report = classification_report(y_test, predictions, output_dict=True)
        
        # Get confusion matrix
        cm = confusion_matrix(y_test, predictions)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': predictions,
            'probabilities': probabilities
        }
        
    def plot_training_history(self, title="MLP Training History"):
        """
        Plot training history.
        
        Args:
            title (str): Plot title
        """
        if self.history is None:
            raise ValueError("Model must be trained before plotting history")
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        ax1.plot(self.history.history['loss'], label='Training Loss')
        ax1.plot(self.history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot accuracy
        ax2.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax2.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
        
    def plot_model_architecture(self, title="MLP Model Architecture"):
        """
        Plot model architecture.
        
        Args:
            title (str): Plot title
        """
        if self.model is None:
            raise ValueError("Model must be built before plotting architecture")
            
        from tensorflow.keras.utils import plot_model
        
        plot_model(self.model, to_file='mlp_architecture.png', 
                  show_shapes=True, show_layer_names=True)
        
        # Display the plot
        img = plt.imread('mlp_architecture.png')
        plt.figure(figsize=(12, 8))
        plt.imshow(img)
        plt.axis('off')
        plt.title(title)
        plt.show()
        
    def get_layer_weights(self, layer_idx):
        """
        Get weights of a specific layer.
        
        Args:
            layer_idx (int): Index of the layer
            
        Returns:
            array: Layer weights
        """
        if self.model is None:
            raise ValueError("Model must be built before accessing weights")
            
        return self.model.layers[layer_idx].get_weights()
        
    def get_model_summary(self):
        """
        Get model summary.
        
        Returns:
            str: Model summary
        """
        if self.model is None:
            raise ValueError("Model must be built before getting summary")
            
        return self.model.summary()
        
    def save_model(self, filepath):
        """
        Save the trained model.
        
        Args:
            filepath (str): Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
            
        self.model.save(filepath)
        
    def load_model(self, filepath):
        """
        Load a trained model.
        
        Args:
            filepath (str): Path to load the model from
        """
        self.model = tf.keras.models.load_model(filepath)
        self.is_fitted = True
