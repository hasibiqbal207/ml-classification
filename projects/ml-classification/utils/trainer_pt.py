"""
PyTorch training utilities.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
from pathlib import Path


class PyTorchTrainer:
    """
    Utility class for training PyTorch models.
    """
    
    def __init__(self, model, model_name="model", device=None):
        """
        Initialize PyTorchTrainer.
        
        Args:
            model: PyTorch model
            model_name (str): Name of the model for saving
            device: Device to use for training ('cpu', 'cuda', or None for auto)
        """
        self.model = model
        self.model_name = model_name
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        self.model.to(self.device)
        
        # Training history
        self.history = {
            'train_losses': [],
            'val_losses': [],
            'train_accuracies': [],
            'val_accuracies': []
        }
        
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              epochs=100, batch_size=32, learning_rate=0.001, 
              optimizer='adam', loss_fn='cross_entropy', verbose=True):
        """
        Train the model.
        
        Args:
            X_train (array-like): Training features
            y_train (array-like): Training labels
            X_val (array-like): Validation features
            y_val (array-like): Validation labels
            epochs (int): Number of epochs
            batch_size (int): Batch size
            learning_rate (float): Learning rate
            optimizer (str): Optimizer name
            loss_fn (str): Loss function name
            verbose (bool): Whether to print progress
            
        Returns:
            dict: Training history
        """
        # Convert data to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.device)
        
        # Create data loader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Set up optimizer
        if optimizer == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        else:
            raise ValueError("Optimizer must be 'adam' or 'sgd'")
            
        # Set up loss function
        if loss_fn == 'cross_entropy':
            self.loss_fn = nn.CrossEntropyLoss()
        elif loss_fn == 'mse':
            self.loss_fn = nn.MSELoss()
        else:
            raise ValueError("Loss function must be 'cross_entropy' or 'mse'")
            
        # Prepare validation data
        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.LongTensor(y_val).to(self.device)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
        # Training loop
        for epoch in range(epochs):
            # Training
            train_loss, train_acc = self._train_epoch(train_loader)
            
            # Validation
            val_loss, val_acc = 0.0, 0.0
            if val_loader is not None:
                val_loss, val_acc = self._validate_epoch(val_loader)
                
            # Store metrics
            self.history['train_losses'].append(train_loss)
            self.history['train_accuracies'].append(train_acc)
            self.history['val_losses'].append(val_loss)
            self.history['val_accuracies'].append(val_acc)
            
            # Print progress
            if verbose and (epoch + 1) % 10 == 0:
                if val_loader is not None:
                    print(f'Epoch [{epoch+1}/{epochs}], '
                          f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                          f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
                else:
                    print(f'Epoch [{epoch+1}/{epochs}], '
                          f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
                    
        return self.history
        
    def _train_epoch(self, train_loader):
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            tuple: (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_X, batch_y in train_loader:
            self.optimizer.zero_grad()
            
            outputs = self.model(batch_X)
            loss = self.loss_fn(outputs, batch_y)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            
        return total_loss / len(train_loader), 100 * correct / total
        
    def _validate_epoch(self, val_loader):
        """
        Validate for one epoch.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            tuple: (average_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = self.model(batch_X)
                loss = self.loss_fn(outputs, batch_y)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
                
        return total_loss / len(val_loader), 100 * correct / total
        
    def evaluate(self, X_test, y_test, batch_size=32):
        """
        Evaluate the model on test data.
        
        Args:
            X_test (array-like): Test features
            y_test (array-like): Test labels
            batch_size (int): Batch size
            
        Returns:
            dict: Evaluation results
        """
        self.model.eval()
        
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        y_test_tensor = torch.LongTensor(y_test).to(self.device)
        
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = self.model(batch_X)
                loss = self.loss_fn(outputs, batch_y)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
                
        accuracy = 100 * correct / total
        avg_loss = total_loss / len(test_loader)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }
        
    def predict(self, X, batch_size=32):
        """
        Make predictions on new data.
        
        Args:
            X (array-like): Input data
            batch_size (int): Batch size
            
        Returns:
            array: Predictions
        """
        self.model.eval()
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        predictions = []
        
        with torch.no_grad():
            for batch_X, in loader:
                outputs = self.model(batch_X)
                _, predicted = torch.max(outputs, 1)
                predictions.extend(predicted.cpu().numpy())
                
        return np.array(predictions)
        
    def predict_proba(self, X, batch_size=32):
        """
        Predict class probabilities.
        
        Args:
            X (array-like): Input data
            batch_size (int): Batch size
            
        Returns:
            array: Class probabilities
        """
        self.model.eval()
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        probabilities = []
        
        with torch.no_grad():
            for batch_X, in loader:
                outputs = self.model(batch_X)
                probs = torch.softmax(outputs, dim=1)
                probabilities.extend(probs.cpu().numpy())
                
        return np.array(probabilities)
        
    def save_model(self, filepath=None):
        """
        Save the trained model.
        
        Args:
            filepath (str): Path to save the model
        """
        if filepath is None:
            filepath = f"models/{self.model_name}_final.pth"
            
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_name': self.model_name,
            'history': self.history
        }, filepath)
        
        print(f"Model saved to {filepath}")
        
    def load_model(self, filepath):
        """
        Load a saved model.
        
        Args:
            filepath (str): Path to the saved model
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint.get('history', {})
        
        print(f"Model loaded from {filepath}")
        
    def plot_training_history(self, figsize=(15, 5)):
        """
        Plot training history.
        
        Args:
            figsize (tuple): Figure size
        """
        if not self.history['train_losses']:
            print("No training history available")
            return
            
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Plot loss
        axes[0].plot(self.history['train_losses'], label='Training Loss')
        if self.history['val_losses']:
            axes[0].plot(self.history['val_losses'], label='Validation Loss')
        axes[0].set_title('Model Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot accuracy
        axes[1].plot(self.history['train_accuracies'], label='Training Accuracy')
        if self.history['val_accuracies']:
            axes[1].plot(self.history['val_accuracies'], label='Validation Accuracy')
        axes[1].set_title('Model Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.show()


def create_pytorch_model(input_size, num_classes, model_type='dense'):
    """
    Create a PyTorch model based on the specified type.
    
    Args:
        input_size (int): Input size
        num_classes (int): Number of output classes
        model_type (str): Type of model ('dense', 'lstm')
        
    Returns:
        nn.Module: Created model
    """
    if model_type == 'dense':
        class DenseModel(nn.Module):
            def __init__(self, input_size, num_classes):
                super(DenseModel, self).__init__()
                self.fc1 = nn.Linear(input_size, 128)
                self.fc2 = nn.Linear(128, 64)
                self.fc3 = nn.Linear(64, num_classes)
                self.dropout = nn.Dropout(0.2)
                self.relu = nn.ReLU()
                
            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.relu(self.fc2(x))
                x = self.dropout(x)
                x = self.fc3(x)
                return x
                
        return DenseModel(input_size, num_classes)
        
    elif model_type == 'lstm':
        class LSTMModel(nn.Module):
            def __init__(self, input_size, num_classes):
                super(LSTMModel, self).__init__()
                self.lstm = nn.LSTM(input_size, 128, batch_first=True)
                self.fc1 = nn.Linear(128, 64)
                self.fc2 = nn.Linear(64, num_classes)
                self.dropout = nn.Dropout(0.2)
                self.relu = nn.ReLU()
                
            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                x = lstm_out[:, -1, :]  # Take the last output
                x = self.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.fc2(x)
                return x
                
        return LSTMModel(input_size, num_classes)
        
    else:
        raise ValueError("Model type must be 'dense' or 'lstm'")
