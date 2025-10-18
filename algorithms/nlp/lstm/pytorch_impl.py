"""
Long Short-Term Memory (LSTM) Implementation using PyTorch
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class LSTMPyTorch(nn.Module):
    """
    LSTM classifier using PyTorch implementation.
    
    This class implements an LSTM network for sequence classification.
    """
    
    def __init__(self, input_size, hidden_size, num_classes, num_layers=2,
                 learning_rate=0.001, random_state=42):
        """
        Initialize the LSTM model.
        
        Args:
            input_size (int): Number of features per time step
            hidden_size (int): Number of LSTM units
            num_classes (int): Number of output classes
            num_layers (int): Number of LSTM layers
            learning_rate (float): Learning rate for optimizer
            random_state (int): Random state for reproducibility
        """
        super(LSTMPyTorch, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.random_state = random_state
        
        # Set random seeds for reproducibility
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(0.3)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)
        
        # Activation function
        self.relu = nn.ReLU()
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()
        
        self.is_fitted = False
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            torch.Tensor: Output tensor
        """
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Take the last output from the sequence
        last_output = lstm_out[:, -1, :]
        
        # Apply batch normalization
        normalized = self.batch_norm(last_output)
        
        # Apply dropout
        dropped = self.dropout(normalized)
        
        # Fully connected layers
        fc1_out = self.relu(self.fc1(dropped))
        fc2_out = self.relu(self.fc2(fc1_out))
        output = self.fc3(fc2_out)
        
        return output
        
    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=100, 
            batch_size=32, verbose=True):
        """
        Fit the LSTM model.
        
        Args:
            X_train (array-like): Training sequences
            y_train (array-like): Training labels
            X_val (array-like): Validation sequences
            y_val (array-like): Validation labels
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            verbose (bool): Whether to print training progress
            
        Returns:
            dict: Training history
        """
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        
        # Create data loader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Prepare validation data
        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val)
            y_val_tensor = torch.LongTensor(y_val)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Training loop
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        
        for epoch in range(epochs):
            # Training
            self.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                
                outputs = self.forward(batch_X)
                loss = self.loss_fn(outputs, batch_y)
                
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
                
            # Validation
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            if val_loader is not None:
                self.eval()
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        outputs = self.forward(batch_X)
                        loss = self.loss_fn(outputs, batch_y)
                        
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += batch_y.size(0)
                        val_correct += (predicted == batch_y).sum().item()
                        
            # Store metrics
            train_losses.append(train_loss / len(train_loader))
            train_accuracies.append(100 * train_correct / train_total)
            
            if val_loader is not None:
                val_losses.append(val_loss / len(val_loader))
                val_accuracies.append(100 * val_correct / val_total)
                
            # Print progress
            if verbose and (epoch + 1) % 10 == 0:
                if val_loader is not None:
                    print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_losses[-1]:.4f}, '
                          f'Train Acc: {train_accuracies[-1]:.2f}%, Val Loss: {val_losses[-1]:.4f}, '
                          f'Val Acc: {val_accuracies[-1]:.2f}%')
                else:
                    print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_losses[-1]:.4f}, '
                          f'Train Acc: {train_accuracies[-1]:.2f}%')
                
        self.is_fitted = True
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies
        }
        
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
            
        self.eval()
        X_tensor = torch.FloatTensor(X)
        
        with torch.no_grad():
            outputs = self.forward(X_tensor)
            _, predictions = torch.max(outputs, 1)
            
        return predictions.numpy()
        
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
            
        self.eval()
        X_tensor = torch.FloatTensor(X)
        
        with torch.no_grad():
            outputs = self.forward(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            
        return probabilities.numpy()
        
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
