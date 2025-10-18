"""
Convolutional Neural Network Implementation using PyTorch
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class CNNPyTorch(nn.Module):
    """
    Convolutional Neural Network classifier using PyTorch implementation.
    
    This class implements a CNN for image classification with configurable architecture.
    """
    
    def __init__(self, input_channels, num_classes, learning_rate=0.001, random_state=42):
        """
        Initialize the CNN model.
        
        Args:
            input_channels (int): Number of input channels
            num_classes (int): Number of output classes
            learning_rate (float): Learning rate for optimizer
            random_state (int): Random state for reproducibility
        """
        super(CNNPyTorch, self).__init__()
        
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.random_state = random_state
        
        # Set random seeds for reproducibility
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        
        # Define the CNN architecture
        self.features = nn.Sequential(
            # First Convolutional Block
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
            
            # Second Convolutional Block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
            
            # Third Convolutional Block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()
        
        self.is_fitted = False
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        x = self.features(x)
        x = self.classifier(x)
        return x
        
    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=100, 
            batch_size=32, verbose=True):
        """
        Fit the CNN model.
        
        Args:
            X_train (array-like): Training images
            y_train (array-like): Training labels
            X_val (array-like): Validation images
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
            X (array-like): Images to predict on
            
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
            X (array-like): Images to predict on
            
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
