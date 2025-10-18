#!/usr/bin/env python3
"""
Data preprocessing script for machine learning project.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
import joblib

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils.data_utils import DataLoader, DataPreprocessor
from utils.config_loader import ConfigLoader


def preprocess_iris_data(input_path="data/raw/iris.csv", output_path="data/processed"):
    """
    Preprocess Iris dataset.
    
    Args:
        input_path (str): Path to input data
        output_path (str): Path to save processed data
    """
    print("Preprocessing Iris dataset...")
    
    # Load data
    df = pd.read_csv(input_path)
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Separate features and target
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    # Encode target labels
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Save processed data
    np.save(f"{output_path}/iris_X.npy", X)
    np.save(f"{output_path}/iris_y.npy", y_encoded)
    
    # Save label encoder
    joblib.dump(le, f"{output_path}/iris_label_encoder.pkl")
    
    print(f"Processed Iris dataset saved to {output_path}/")


def preprocess_wine_data(input_path="data/raw/wine.csv", output_path="data/processed"):
    """
    Preprocess Wine dataset.
    
    Args:
        input_path (str): Path to input data
        output_path (str): Path to save processed data
    """
    print("Preprocessing Wine dataset...")
    
    # Load data
    df = pd.read_csv(input_path)
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Separate features and target
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    # Encode target labels
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Save processed data
    np.save(f"{output_path}/wine_X.npy", X)
    np.save(f"{output_path}/wine_y.npy", y_encoded)
    
    # Save label encoder
    joblib.dump(le, f"{output_path}/wine_label_encoder.pkl")
    
    print(f"Processed Wine dataset saved to {output_path}/")


def preprocess_breast_cancer_data(input_path="data/raw/breast_cancer.csv", output_path="data/processed"):
    """
    Preprocess Breast Cancer dataset.
    
    Args:
        input_path (str): Path to input data
        output_path (str): Path to save processed data
    """
    print("Preprocessing Breast Cancer dataset...")
    
    # Load data
    df = pd.read_csv(input_path)
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Separate features and target
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    # Encode target labels
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Save processed data
    np.save(f"{output_path}/breast_cancer_X.npy", X)
    np.save(f"{output_path}/breast_cancer_y.npy", y_encoded)
    
    # Save label encoder
    joblib.dump(le, f"{output_path}/breast_cancer_label_encoder.pkl")
    
    print(f"Processed Breast Cancer dataset saved to {output_path}/")


def preprocess_mnist_data(input_path="data/raw", output_path="data/processed"):
    """
    Preprocess MNIST dataset.
    
    Args:
        input_path (str): Path to input data
        output_path (str): Path to save processed data
    """
    print("Preprocessing MNIST dataset...")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Load data
    X = np.load(f"{input_path}/mnist_images.npy")
    y = np.load(f"{input_path}/mnist_labels.npy")
    
    # Normalize pixel values
    X = X.astype(np.float32) / 255.0
    
    # Reshape for CNN (if needed)
    X_cnn = X.reshape(-1, 28, 28, 1)
    
    # Save processed data
    np.save(f"{output_path}/mnist_X.npy", X)
    np.save(f"{output_path}/mnist_X_cnn.npy", X_cnn)
    np.save(f"{output_path}/mnist_y.npy", y)
    
    print(f"Processed MNIST dataset saved to {output_path}/")


def preprocess_cifar10_data(input_path="data/raw", output_path="data/processed"):
    """
    Preprocess CIFAR-10 dataset.
    
    Args:
        input_path (str): Path to input data
        output_path (str): Path to save processed data
    """
    print("Preprocessing CIFAR-10 dataset...")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Load data
    X = np.load(f"{input_path}/cifar10_images.npy")
    y = np.load(f"{input_path}/cifar10_labels.npy")
    
    # Normalize pixel values
    X = X.astype(np.float32) / 255.0
    
    # Reshape for CNN
    X_cnn = X.reshape(-1, 32, 32, 3)
    
    # Save processed data
    np.save(f"{output_path}/cifar10_X.npy", X)
    np.save(f"{output_path}/cifar10_X_cnn.npy", X_cnn)
    np.save(f"{output_path}/cifar10_y.npy", y)
    
    print(f"Processed CIFAR-10 dataset saved to {output_path}/")


def preprocess_custom_data(input_path, output_path="data/processed", config_path="config.yaml"):
    """
    Preprocess custom dataset using configuration.
    
    Args:
        input_path (str): Path to input data
        output_path (str): Path to save processed data
        config_path (str): Path to configuration file
    """
    print("Preprocessing custom dataset...")
    
    # Load configuration
    config_loader = ConfigLoader(config_path)
    config_loader.load_config()
    
    # Load data
    data_loader = DataLoader()
    df = data_loader.load_csv(input_path)
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Get preprocessing configuration
    preprocess_config = config_loader.get('data.preprocessing', {})
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Separate features and target (assume last column is target)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    # Preprocess data
    processed_data = preprocessor.preprocess_pipeline(
        X, y,
        handle_missing=preprocess_config.get('handle_missing', True),
        encode_categorical=preprocess_config.get('encode_categorical', True),
        scale_features=preprocess_config.get('scale_features', True),
        split_data=False
    )
    
    # Save processed data
    np.save(f"{output_path}/custom_X.npy", processed_data['X'])
    np.save(f"{output_path}/custom_y.npy", y)
    
    print(f"Processed custom dataset saved to {output_path}/")


def main():
    """
    Main function for command-line interface.
    """
    parser = argparse.ArgumentParser(description='Preprocess datasets for ML project')
    parser.add_argument('--dataset', type=str, required=True,
                      choices=['iris', 'wine', 'breast_cancer', 'mnist', 'cifar10', 'custom'],
                      help='Dataset to preprocess')
    parser.add_argument('--input_path', type=str,
                      help='Input path for data')
    parser.add_argument('--output_path', type=str, default='data/processed',
                      help='Output path for processed data')
    parser.add_argument('--config', type=str, default='config.yaml',
                      help='Configuration file path')
    
    args = parser.parse_args()
    
    # Set default input paths
    if args.input_path is None:
        if args.dataset == 'iris':
            args.input_path = "data/raw/iris.csv"
        elif args.dataset == 'wine':
            args.input_path = "data/raw/wine.csv"
        elif args.dataset == 'breast_cancer':
            args.input_path = "data/raw/breast_cancer.csv"
        elif args.dataset == 'mnist':
            args.input_path = "data/raw"
        elif args.dataset == 'cifar10':
            args.input_path = "data/raw"
        else:
            print("Error: --input_path is required for custom dataset")
            return
    
    if args.dataset == 'iris':
        preprocess_iris_data(args.input_path, args.output_path)
    elif args.dataset == 'wine':
        preprocess_wine_data(args.input_path, args.output_path)
    elif args.dataset == 'breast_cancer':
        preprocess_breast_cancer_data(args.input_path, args.output_path)
    elif args.dataset == 'mnist':
        preprocess_mnist_data(args.input_path, args.output_path)
    elif args.dataset == 'cifar10':
        preprocess_cifar10_data(args.input_path, args.output_path)
    elif args.dataset == 'custom':
        preprocess_custom_data(args.input_path, args.output_path, args.config)
    
    print("Preprocessing completed!")


if __name__ == "__main__":
    main()
