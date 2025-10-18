"""
Logistic Regression Implementation using PyTorch
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class LogisticRegressionPyTorch(nn.Module):
    """
    Logistic Regression classifier using PyTorch implementation.
    
    This class implements logistic regression as a neural network with
    a single linear layer and sigmoid/softmax activation.
    """
    
    def __init__(self, input_dim, num_classes, learning_rate=0.01, random_state=42):
        """
        Initialize the Logistic Regression model.
        
        Args:
            input_dim (int): Number of input features
            num_classes (int): Number of output classes
            learning_rate (float): Learning rate for optimizer
            random_state (int): Random state for reproducibility
        """
        super(LogisticRegressionPyTorch, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.random_state = random_state
        
        # Set random seeds for reproducibility
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        
        # Define the linear layer
        self.linear = nn.Linear(input_dim, num_classes)
        
        # Define activation function
        if num_classes == 2:
            self.activation = nn.Sigmoid()
            self.loss_fn = nn.BCELoss()
        else:
            self.activation = nn.Softmax(dim=1)
            self.loss_fn = nn.CrossEntropyLoss()
            
        # Initialize optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        # Scaler for preprocessing
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        x = self.linear(x)
        if self.num_classes == 2:
            return self.activation(x)
        else:
            return self.activation(x)
            
    def fit(self, X, y, epochs=100, batch_size=32, validation_split=0.2, verbose=True):
        """
        Fit the logistic regression model.
        
        Args:
            X (array-like): Training features
            y (array-like): Training labels
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            validation_split (float): Fraction of data to use for validation
            verbose (bool): Whether to print training progress
        """
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data for validation
        split_idx = int(len(X_scaled) * (1 - validation_split))
        X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train) if self.num_classes == 2 else torch.LongTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val) if self.num_classes == 2 else torch.LongTensor(y_val)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Training loop
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training
            self.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                
                outputs = self.forward(batch_X)
                
                if self.num_classes == 2:
                    loss = self.loss_fn(outputs.squeeze(), batch_y)
                else:
                    loss = self.loss_fn(outputs, batch_y)
                    
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                
            # Validation
            self.eval()
            with torch.no_grad():
                val_outputs = self.forward(X_val_tensor)
                if self.num_classes == 2:
                    val_loss = self.loss_fn(val_outputs.squeeze(), y_val_tensor).item()
                else:
                    val_loss = self.loss_fn(val_outputs, y_val_tensor).item()
                    
            train_losses.append(train_loss / len(train_loader))
            val_losses.append(val_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}')
                
        self.is_fitted = True
        return {'train_losses': train_losses, 'val_losses': val_losses}
        
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
            
        self.eval()
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled)
        
        with torch.no_grad():
            outputs = self.forward(X_tensor)
            
            if self.num_classes == 2:
                predictions = (outputs > 0.5).float().numpy().flatten()
            else:
                predictions = torch.argmax(outputs, dim=1).numpy()
                
        return predictions
        
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
            
        self.eval()
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled)
        
        with torch.no_grad():
            outputs = self.forward(X_tensor)
            probabilities = outputs.numpy()
            
        return probabilities
        
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
