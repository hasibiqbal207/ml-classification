"""
Data utilities for loading, preprocessing, and managing datasets.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
import pickle
import joblib


class DataLoader:
    """
    Utility class for loading and managing datasets.
    """
    
    def __init__(self, data_path="data"):
        """
        Initialize DataLoader.
        
        Args:
            data_path (str): Path to data directory
        """
        self.data_path = Path(data_path)
        self.raw_path = self.data_path / "raw"
        self.processed_path = self.data_path / "processed"
        
    def load_csv(self, filename, **kwargs):
        """
        Load CSV file.
        
        Args:
            filename (str): Name of CSV file
            **kwargs: Additional arguments for pd.read_csv
            
        Returns:
            pd.DataFrame: Loaded data
        """
        file_path = self.raw_path / filename
        return pd.read_csv(file_path, **kwargs)
        
    def load_excel(self, filename, **kwargs):
        """
        Load Excel file.
        
        Args:
            filename (str): Name of Excel file
            **kwargs: Additional arguments for pd.read_excel
            
        Returns:
            pd.DataFrame: Loaded data
        """
        file_path = self.raw_path / filename
        return pd.read_excel(file_path, **kwargs)
        
    def save_processed(self, data, filename):
        """
        Save processed data.
        
        Args:
            data: Data to save (DataFrame, array, etc.)
            filename (str): Name of file to save
        """
        file_path = self.processed_path / filename
        
        if isinstance(data, pd.DataFrame):
            data.to_csv(file_path, index=False)
        elif isinstance(data, np.ndarray):
            np.save(file_path, data)
        else:
            # Save as pickle for other data types
            with open(file_path.with_suffix('.pkl'), 'wb') as f:
                pickle.dump(data, f)
                
    def load_processed(self, filename):
        """
        Load processed data.
        
        Args:
            filename (str): Name of file to load
            
        Returns:
            Loaded data
        """
        file_path = self.processed_path / filename
        
        if file_path.suffix == '.csv':
            return pd.read_csv(file_path)
        elif file_path.suffix == '.npy':
            return np.load(file_path)
        else:
            # Load from pickle
            with open(file_path.with_suffix('.pkl'), 'rb') as f:
                return pickle.load(f)


class DataPreprocessor:
    """
    Utility class for data preprocessing operations.
    """
    
    def __init__(self):
        """
        Initialize DataPreprocessor.
        """
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.onehot_encoder = OneHotEncoder()
        self.imputer = SimpleImputer()
        
    def handle_missing_values(self, X, strategy='mean'):
        """
        Handle missing values in the dataset.
        
        Args:
            X (array-like): Input data
            strategy (str): Strategy for imputation ('mean', 'median', 'most_frequent')
            
        Returns:
            array: Data with missing values imputed
        """
        self.imputer = SimpleImputer(strategy=strategy)
        return self.imputer.fit_transform(X)
        
    def encode_categorical(self, X, method='label'):
        """
        Encode categorical variables.
        
        Args:
            X (array-like): Categorical data
            method (str): Encoding method ('label', 'onehot')
            
        Returns:
            array: Encoded data
        """
        if method == 'label':
            return self.label_encoder.fit_transform(X)
        elif method == 'onehot':
            return self.onehot_encoder.fit_transform(X).toarray()
        else:
            raise ValueError("Method must be 'label' or 'onehot'")
            
    def scale_features(self, X, fit=True):
        """
        Scale features using standardization.
        
        Args:
            X (array-like): Input features
            fit (bool): Whether to fit the scaler
            
        Returns:
            array: Scaled features
        """
        if fit:
            return self.scaler.fit_transform(X)
        else:
            return self.scaler.transform(X)
            
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """
        Split data into train and test sets.
        
        Args:
            X (array-like): Features
            y (array-like): Labels
            test_size (float): Proportion of test set
            random_state (int): Random seed
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
        
    def preprocess_pipeline(self, X, y=None, handle_missing=True, 
                          encode_categorical=True, scale_features=True,
                          split_data=True, test_size=0.2):
        """
        Complete preprocessing pipeline.
        
        Args:
            X (array-like): Features
            y (array-like): Labels (optional)
            handle_missing (bool): Whether to handle missing values
            encode_categorical (bool): Whether to encode categorical variables
            scale_features (bool): Whether to scale features
            split_data (bool): Whether to split data
            test_size (float): Proportion of test set
            
        Returns:
            dict: Preprocessed data
        """
        result = {}
        
        # Handle missing values
        if handle_missing:
            X = self.handle_missing_values(X)
            
        # Encode categorical variables
        if encode_categorical:
            # This is a simplified version - in practice, you'd need to identify categorical columns
            pass
            
        # Scale features
        if scale_features:
            X = self.scale_features(X, fit=True)
            
        # Split data
        if split_data and y is not None:
            X_train, X_test, y_train, y_test = self.split_data(X, y, test_size)
            result = {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test
            }
        else:
            result = {'X': X}
            
        return result


def load_dataset(dataset_name, data_path="data"):
    """
    Load common datasets.
    
    Args:
        dataset_name (str): Name of dataset ('iris', 'wine', 'breast_cancer')
        data_path (str): Path to data directory
        
    Returns:
        tuple: (X, y, feature_names, target_names)
    """
    from sklearn.datasets import load_iris, load_wine, load_breast_cancer
    
    datasets = {
        'iris': load_iris,
        'wine': load_wine,
        'breast_cancer': load_breast_cancer
    }
    
    if dataset_name not in datasets:
        raise ValueError(f"Dataset {dataset_name} not supported")
        
    data = datasets[dataset_name]()
    return data.data, data.target, data.feature_names, data.target_names
