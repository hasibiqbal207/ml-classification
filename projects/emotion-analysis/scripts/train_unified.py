#!/usr/bin/env python3
"""
Unified Model Training Framework for GoEmotions Multilabel Classification

This script provides a unified interface to train any available algorithm
using scikit-learn's multilabel classification capabilities.
"""

import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss, jaccard_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class UniversalModelTrainer:
    """
    Universal trainer that can train any available algorithm for multilabel classification.
    """
    
    def __init__(self, data_dir: str = None, results_dir: str = None):
        """
        Initialize the universal trainer.
        
        Args:
            data_dir: Path to data directory
            results_dir: Path to results directory
        """
        if data_dir is None:
            self.data_dir = Path(__file__).parent.parent / "data"
        else:
            self.data_dir = Path(data_dir)
            
        if results_dir is None:
            self.results_dir = Path(__file__).parent.parent / "results"
        else:
            self.results_dir = Path(results_dir)
            
        # Create results directory
        self.results_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for organized storage
        (self.results_dir / "models").mkdir(exist_ok=True)
        (self.results_dir / "visualizations").mkdir(exist_ok=True)
        (self.results_dir / "reports").mkdir(exist_ok=True)
        
        # Available algorithms mapping for multilabel classification
        self.available_algorithms = {
            'naive_bayes': {
                'class': MultinomialNB,
                'default_params': {'alpha': 1.0},
                'vectorizer': 'CountVectorizer',
                'multilabel_wrapper': OneVsRestClassifier
            },
            'logistic_regression': {
                'class': LogisticRegression,
                'default_params': {'random_state': 42, 'max_iter': 1000},
                'vectorizer': 'TfidfVectorizer',
                'multilabel_wrapper': OneVsRestClassifier
            },
            'random_forest': {
                'class': RandomForestClassifier,
                'default_params': {'n_estimators': 100, 'random_state': 42},
                'vectorizer': 'TfidfVectorizer',
                'multilabel_wrapper': OneVsRestClassifier
            },
            'svm': {
                'class': SVC,
                'default_params': {'kernel': 'linear', 'random_state': 42, 'probability': True},
                'vectorizer': 'TfidfVectorizer',
                'multilabel_wrapper': OneVsRestClassifier
            },
            'knn': {
                'class': KNeighborsClassifier,
                'default_params': {'n_neighbors': 5},
                'vectorizer': 'TfidfVectorizer',
                'multilabel_wrapper': OneVsRestClassifier
            },
            'adaboost': {
                'class': AdaBoostClassifier,
                'default_params': {'n_estimators': 50, 'random_state': 42},
                'vectorizer': 'TfidfVectorizer',
                'multilabel_wrapper': OneVsRestClassifier
            },
            'decision_tree': {
                'class': DecisionTreeClassifier,
                'default_params': {'random_state': 42},
                'vectorizer': 'TfidfVectorizer',
                'multilabel_wrapper': OneVsRestClassifier
            },
        }
        
        self.model = None
        self.vectorizer = None
        self.emotion_categories = None
        self.algorithm_name = None
        
    def load_data(self) -> tuple:
        """
        Load processed training data.
        
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test, emotion_categories)
        """
        logger.info("Loading processed data...")
        
        # Load text data
        train_texts_path = self.data_dir / "processed" / "train_texts.csv"
        val_texts_path = self.data_dir / "processed" / "val_texts.csv"
        test_texts_path = self.data_dir / "processed" / "test_texts.csv"
        
        # Load label data
        train_labels_path = self.data_dir / "processed" / "train_labels.csv"
        val_labels_path = self.data_dir / "processed" / "val_labels.csv"
        test_labels_path = self.data_dir / "processed" / "test_labels.csv"
        
        # Load emotion categories
        categories_path = self.data_dir / "processed" / "emotion_categories.pkl"
        
        if not all([train_texts_path.exists(), val_texts_path.exists(), test_texts_path.exists(),
                   train_labels_path.exists(), val_labels_path.exists(), test_labels_path.exists(),
                   categories_path.exists()]):
            raise FileNotFoundError("Processed data files not found. Run preprocess_data.py first.")
        
        # Load text data
        X_train = pd.read_csv(train_texts_path)['text'].values
        X_val = pd.read_csv(val_texts_path)['text'].values
        X_test = pd.read_csv(test_texts_path)['text'].values
        
        # Load label data
        y_train = pd.read_csv(train_labels_path).values
        y_val = pd.read_csv(val_labels_path).values
        y_test = pd.read_csv(test_labels_path).values
        
        # Load emotion categories
        with open(categories_path, 'rb') as f:
            self.emotion_categories = pickle.load(f)
        
        logger.info(f"Loaded data: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        logger.info(f"Number of emotion categories: {len(self.emotion_categories)}")
        logger.info(f"Label matrix shape: {y_train.shape}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test, self.emotion_categories
    
    def create_model(self, algorithm_name: str, **kwargs) -> None:
        """
        Create a model instance using the appropriate algorithm class.
        
        Args:
            algorithm_name: Name of the algorithm to use
            **kwargs: Additional parameters for the algorithm
        """
        if algorithm_name not in self.available_algorithms:
            available = ', '.join(self.available_algorithms.keys())
            raise ValueError(f"Algorithm '{algorithm_name}' not available. Available: {available}")
        
        logger.info(f"Creating {algorithm_name} model...")
        
        # Get algorithm configuration
        algo_config = self.available_algorithms[algorithm_name]
        
        # Merge default parameters with provided parameters
        params = {**algo_config['default_params'], **kwargs}
        
        # Create base model
        base_model = algo_config['class'](**params)
        
        # Wrap with multilabel classifier
        self.model = algo_config['multilabel_wrapper'](base_model)
        self.algorithm_name = algorithm_name
        
        # Create appropriate vectorizer
        vectorizer_class = algo_config['vectorizer']
        if vectorizer_class == 'CountVectorizer':
            self.vectorizer = CountVectorizer(max_features=10000, stop_words='english')
        elif vectorizer_class == 'TfidfVectorizer':
            self.vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
        else:
            raise ValueError(f"Unknown vectorizer: {vectorizer_class}")
        
        logger.info(f"Created {algorithm_name} model with parameters: {params}")
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                   X_val: np.ndarray = None, y_val: np.ndarray = None,
                   hyperparameter_tuning: bool = False) -> Dict[str, Any]:
        """
        Train the model using multilabel classification.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            hyperparameter_tuning: Whether to perform hyperparameter tuning
            
        Returns:
            dict: Training results and metrics
        """
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")
        
        logger.info(f"Training {self.algorithm_name} model...")
        
        # Transform text data using vectorizer
        logger.info("Transforming text data...")
        X_train_transformed = self.vectorizer.fit_transform(X_train)
        
        if X_val is not None:
            X_val_transformed = self.vectorizer.transform(X_val)
        
        # Train the model
        logger.info("Fitting model...")
        self.model.fit(X_train_transformed, y_train)
        
        # Evaluate on validation set if provided
        results = {}
        if X_val is not None:
            logger.info("Evaluating on validation set...")
            val_results = self.evaluate_multilabel(X_val_transformed, y_val)
            results['validation'] = val_results
        
        logger.info(f"{self.algorithm_name} model training completed!")
        return results
    
    def evaluate_multilabel(self, X: np.ndarray, y_true: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate multilabel model performance.
        
        Args:
            X: Features
            y_true: True labels
            
        Returns:
            dict: Evaluation metrics
        """
        # Make predictions
        y_pred = self.model.predict(X)
        
        # Calculate multilabel metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'hamming_loss': hamming_loss(y_true, y_pred),
            'jaccard_score': jaccard_score(y_true, y_pred, average='samples'),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_micro': precision_score(y_true, y_pred, average='micro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_micro': recall_score(y_true, y_pred, average='micro', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=0),
        }
        
        return metrics
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate the model using multilabel evaluation methods.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            dict: Evaluation results
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        logger.info(f"Evaluating {self.algorithm_name} model on test set...")
        
        # Transform test data
        X_test_transformed = self.vectorizer.transform(X_test)
        
        # Evaluate using multilabel metrics
        test_results = self.evaluate_multilabel(X_test_transformed, y_test)
        
        # Generate per-emotion metrics
        y_pred = self.model.predict(X_test_transformed)
        per_emotion_metrics = {}
        
        for i, emotion in enumerate(self.emotion_categories):
            per_emotion_metrics[emotion] = {
                'precision': precision_score(y_test[:, i], y_pred[:, i], zero_division=0),
                'recall': recall_score(y_test[:, i], y_pred[:, i], zero_division=0),
                'f1': f1_score(y_test[:, i], y_pred[:, i], zero_division=0)
            }
        
        test_results['per_emotion'] = per_emotion_metrics
        
        logger.info(f"{self.algorithm_name} model evaluation completed!")
        return test_results
    
    def save_model(self) -> None:
        """
        Save the trained model and vectorizer.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        logger.info(f"Saving {self.algorithm_name} model...")
        
        # Save model
        model_path = self.results_dir / "models" / f"{self.algorithm_name}_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save vectorizer
        vectorizer_path = self.results_dir / "models" / f"{self.algorithm_name}_vectorizer.pkl"
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        # Save emotion categories
        categories_path = self.results_dir / "models" / f"{self.algorithm_name}_emotion_categories.pkl"
        with open(categories_path, 'wb') as f:
            pickle.dump(self.emotion_categories, f)
        
        logger.info(f"Model, vectorizer, and emotion categories saved to {self.results_dir}")
    
    def train_and_evaluate(self, algorithm_name: str, hyperparameter_tuning: bool = False, **kwargs) -> Dict[str, Any]:
        """
        Complete training and evaluation pipeline.
        
        Args:
            algorithm_name: Name of the algorithm to train
            hyperparameter_tuning: Whether to perform hyperparameter tuning
            **kwargs: Additional parameters for the algorithm
            
        Returns:
            dict: Complete results including training and evaluation metrics
        """
        logger.info(f"Starting complete training pipeline for {algorithm_name}")
        
        # Load data
        X_train, X_val, X_test, y_train, y_val, y_test, emotion_categories = self.load_data()
        
        # Create model
        self.create_model(algorithm_name, **kwargs)
        
        # Train model
        train_results = self.train_model(X_train, y_train, X_val, y_val, hyperparameter_tuning)
        
        # Evaluate model
        test_results = self.evaluate_model(X_test, y_test)
        
        # Save model
        self.save_model()
        
        # Compile results
        results = {
            'algorithm': algorithm_name,
            'training': train_results,
            'test': test_results,
            'model_path': str(self.results_dir / "models" / f"{algorithm_name}_model.pkl"),
            'vectorizer_path': str(self.results_dir / "models" / f"{algorithm_name}_vectorizer.pkl"),
            'categories_path': str(self.results_dir / "models" / f"{algorithm_name}_emotion_categories.pkl")
        }
        
        logger.info(f"Complete pipeline finished for {algorithm_name}")
        return results


def main():
    """Main function to run the universal trainer."""
    parser = argparse.ArgumentParser(description='Universal Model Trainer for GoEmotions Multilabel Classification')
    parser.add_argument('--algorithm', required=True, 
                       help='Algorithm to train (naive_bayes, logistic_regression, random_forest, svm, knn, adaboost, decision_tree)')
    parser.add_argument('--data-dir', help='Path to data directory')
    parser.add_argument('--results-dir', help='Path to results directory')
    parser.add_argument('--tuning', action='store_true', help='Perform hyperparameter tuning')
    parser.add_argument('--params', nargs='*', help='Additional parameters as key=value pairs')
    
    args = parser.parse_args()
    
    # Parse additional parameters
    extra_params = {}
    if args.params:
        for param in args.params:
            key, value = param.split('=')
            # Try to convert to appropriate type
            try:
                if value.lower() == 'true':
                    extra_params[key] = True
                elif value.lower() == 'false':
                    extra_params[key] = False
                elif value.isdigit():
                    extra_params[key] = int(value)
                else:
                    try:
                        extra_params[key] = float(value)
                    except ValueError:
                        extra_params[key] = value
            except ValueError:
                extra_params[key] = value
    
    # Create trainer
    trainer = UniversalModelTrainer(args.data_dir, args.results_dir)
    
    # Train and evaluate
    try:
        results = trainer.train_and_evaluate(
            algorithm_name=args.algorithm,
            hyperparameter_tuning=args.tuning,
            **extra_params
        )
        
        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETED FOR {args.algorithm.upper()}")
        print(f"{'='*60}")
        
        if 'test' in results:
            test_metrics = results['test']
            accuracy = test_metrics.get('accuracy', 'N/A')
            hamming_loss = test_metrics.get('hamming_loss', 'N/A')
            f1_macro = test_metrics.get('f1_macro', 'N/A')
            f1_micro = test_metrics.get('f1_micro', 'N/A')
            
            print(f"Test Accuracy: {accuracy:.4f}" if isinstance(accuracy, (int, float)) else f"Test Accuracy: {accuracy}")
            print(f"Hamming Loss: {hamming_loss:.4f}" if isinstance(hamming_loss, (int, float)) else f"Hamming Loss: {hamming_loss}")
            print(f"F1-Score (Macro): {f1_macro:.4f}" if isinstance(f1_macro, (int, float)) else f"F1-Score (Macro): {f1_macro}")
            print(f"F1-Score (Micro): {f1_micro:.4f}" if isinstance(f1_micro, (int, float)) else f"F1-Score (Micro): {f1_micro}")
        
        print(f"\nModel saved to: {results['model_path']}")
        print(f"Vectorizer saved to: {results['vectorizer_path']}")
        print(f"Emotion categories saved to: {results['categories_path']}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        print(f"❌ Training failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())