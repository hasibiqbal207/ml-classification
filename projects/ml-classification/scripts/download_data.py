#!/usr/bin/env python3
"""
Data download script for machine learning project.
"""

import argparse
import requests
import os
from pathlib import Path
import pandas as pd
from sklearn.datasets import fetch_openml
import zipfile
import tarfile


def download_iris_dataset(output_path="data/raw"):
    """
    Download Iris dataset.
    
    Args:
        output_path (str): Path to save the dataset
    """
    print("Downloading Iris dataset...")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Load Iris dataset
    iris = fetch_openml('iris', version=1, as_frame=True)
    df = iris.frame
    
    # Save to CSV
    df.to_csv(f"{output_path}/iris.csv", index=False)
    print(f"Iris dataset saved to {output_path}/iris.csv")


def download_wine_dataset(output_path="data/raw"):
    """
    Download Wine dataset.
    
    Args:
        output_path (str): Path to save the dataset
    """
    print("Downloading Wine dataset...")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Load Wine dataset
    wine = fetch_openml('wine', version=1, as_frame=True)
    df = wine.frame
    
    # Save to CSV
    df.to_csv(f"{output_path}/wine.csv", index=False)
    print(f"Wine dataset saved to {output_path}/wine.csv")


def download_breast_cancer_dataset(output_path="data/raw"):
    """
    Download Breast Cancer dataset.
    
    Args:
        output_path (str): Path to save the dataset
    """
    print("Downloading Breast Cancer dataset...")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Load Breast Cancer dataset
    cancer = fetch_openml('breast-cancer', version=1, as_frame=True)
    df = cancer.frame
    
    # Save to CSV
    df.to_csv(f"{output_path}/breast_cancer.csv", index=False)
    print(f"Breast Cancer dataset saved to {output_path}/breast_cancer.csv")


def download_mnist_dataset(output_path="data/raw"):
    """
    Download MNIST dataset.
    
    Args:
        output_path (str): Path to save the dataset
    """
    print("Downloading MNIST dataset...")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Load MNIST dataset
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    
    # Save as numpy arrays
    import numpy as np
    np.save(f"{output_path}/mnist_images.npy", mnist.data)
    np.save(f"{output_path}/mnist_labels.npy", mnist.target.astype(int))
    
    print(f"MNIST dataset saved to {output_path}/")


def download_cifar10_dataset(output_path="data/raw"):
    """
    Download CIFAR-10 dataset.
    
    Args:
        output_path (str): Path to save the dataset
    """
    print("Downloading CIFAR-10 dataset...")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Load CIFAR-10 dataset
    cifar10 = fetch_openml('CIFAR_10', version=1, as_frame=False)
    
    # Save as numpy arrays
    import numpy as np
    np.save(f"{output_path}/cifar10_images.npy", cifar10.data)
    np.save(f"{output_path}/cifar10_labels.npy", cifar10.target.astype(int))
    
    print(f"CIFAR-10 dataset saved to {output_path}/")


def download_custom_dataset(url, output_path="data/raw", filename=None):
    """
    Download a custom dataset from URL.
    
    Args:
        url (str): URL to download from
        output_path (str): Path to save the dataset
        filename (str): Name of the file (if None, extract from URL)
    """
    print(f"Downloading dataset from {url}...")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Get filename from URL if not provided
    if filename is None:
        filename = url.split('/')[-1]
    
    # Download file
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    file_path = f"{output_path}/{filename}"
    
    with open(file_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    print(f"Dataset saved to {file_path}")
    
    # Extract if it's a compressed file
    if filename.endswith('.zip'):
        print("Extracting ZIP file...")
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(output_path)
        print("Extraction completed")
    elif filename.endswith(('.tar.gz', '.tgz')):
        print("Extracting TAR file...")
        with tarfile.open(file_path, 'r:gz') as tar_ref:
            tar_ref.extractall(output_path)
        print("Extraction completed")


def main():
    """
    Main function for command-line interface.
    """
    parser = argparse.ArgumentParser(description='Download datasets for ML project')
    parser.add_argument('--dataset', type=str, required=True,
                      choices=['iris', 'wine', 'breast_cancer', 'mnist', 'cifar10', 'custom'],
                      help='Dataset to download')
    parser.add_argument('--output_path', type=str, default='data/raw',
                      help='Output path for downloaded data')
    parser.add_argument('--url', type=str,
                      help='URL for custom dataset (required for custom)')
    parser.add_argument('--filename', type=str,
                      help='Filename for custom dataset')
    
    args = parser.parse_args()
    
    if args.dataset == 'iris':
        download_iris_dataset(args.output_path)
    elif args.dataset == 'wine':
        download_wine_dataset(args.output_path)
    elif args.dataset == 'breast_cancer':
        download_breast_cancer_dataset(args.output_path)
    elif args.dataset == 'mnist':
        download_mnist_dataset(args.output_path)
    elif args.dataset == 'cifar10':
        download_cifar10_dataset(args.output_path)
    elif args.dataset == 'custom':
        if not args.url:
            print("Error: --url is required for custom dataset")
            return
        download_custom_dataset(args.url, args.output_path, args.filename)
    
    print("Download completed!")


if __name__ == "__main__":
    main()
