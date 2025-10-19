#!/usr/bin/env python3
"""
Unified Model Training Framework for 20 Newsgroups Classification

This script provides a unified interface to train any available algorithm
using the existing algorithm classes from the algorithms/ folder.
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
from sklearn.preprocessing import LabelEncoder
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
    Universal trainer that can train any available algorithm using the existing algorithm classes.
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
        
        # Available algorithms mapping
        self.available_algorithms = {
            'naive_bayes': {
                'class': 'NaiveBayesSklearn',
                'module': 'algorithms.naive_bayes.sklearn_impl',
                'default_params': {'variant': 'multinomial', 'alpha': 1.0},
                'vectorizer': 'CountVectorizer'
            },
            'logistic_regression': {
                'class': 'LogisticRegressionSklearn',
                'module': 'algorithms.logistic_regression.sklearn_impl',
                'default_params': {'random_state': 42, 'max_iter': 1000},
                'vectorizer': 'TfidfVectorizer'
            },
            'random_forest': {
                'class': 'RandomForestSklearn',
                'module': 'algorithms.random_forest.sklearn_impl',
                'default_params': {'n_estimators': 100, 'random_state': 42},
                'vectorizer': 'TfidfVectorizer'
            },
            'svm': {
                'class': 'SVMSklearn',
                'module': 'algorithms.svm.sklearn_impl',
                'default_params': {'kernel': 'linear', 'random_state': 42},
                'vectorizer': 'TfidfVectorizer'
            },
            'knn': {
                'class': 'KNNSklearn',
                'module': 'algorithms.knn.sklearn_impl',
                'default_params': {'n_neighbors': 5},
                'vectorizer': 'TfidfVectorizer'
            },
            'adaboost': {
                'class': 'AdaBoostSklearn',
                'module': 'algorithms.adaboost.sklearn_impl',
                'default_params': {'n_estimators': 50, 'random_state': 42},
                'vectorizer': 'TfidfVectorizer'
            },
            'decision_tree': {
                'class': 'DecisionTreeSklearn',
                'module': 'algorithms.decision_tree.sklearn_impl',
                'default_params': {'random_state': 42},
                'vectorizer': 'TfidfVectorizer'
            },
        }
        
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        self.algorithm_name = None
        
    def load_data(self) -> tuple:
        """
        Load processed training data.
        
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        logger.info("Loading processed data...")
        
        # Load training data
        train_path = self.data_dir / "processed" / "train.csv"
        val_path = self.data_dir / "processed" / "val.csv"
        test_path = self.data_dir / "processed" / "test.csv"
        
        if not all([train_path.exists(), val_path.exists(), test_path.exists()]):
            raise FileNotFoundError("Processed data files not found. Run preprocess_data.py first.")
        
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)
        
        # Extract features and labels
        X_train = train_df['text'].values
        y_train = train_df['label'].values
        X_val = val_df['text'].values
        y_val = val_df['label'].values
        X_test = test_df['text'].values
        y_test = test_df['label'].values
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_train = self.label_encoder.fit_transform(y_train)
        y_val = self.label_encoder.transform(y_val)
        y_test = self.label_encoder.transform(y_test)
        
        logger.info(f"Loaded data: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        logger.info(f"Number of classes: {len(self.label_encoder.classes_)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
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
        
        # Import the algorithm class
        module = __import__(algo_config['module'], fromlist=[algo_config['class']])
        algorithm_class = getattr(module, algo_config['class'])
        
        # Merge default parameters with provided parameters
        params = {**algo_config['default_params'], **kwargs}
        
        # Create model instance
        self.model = algorithm_class(**params)
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
        Train the model using the algorithm class's built-in methods.
        
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
        
        # Train the model using the algorithm class's fit method
        logger.info("Fitting model...")
        self.model.fit(X_train_transformed.toarray(), y_train)
        
        # Perform hyperparameter tuning if requested
        if hyperparameter_tuning and hasattr(self.model, 'hyperparameter_tuning'):
            logger.info("Performing hyperparameter tuning...")
            self.model.hyperparameter_tuning(X_train_transformed.toarray(), y_train)
        
        # Evaluate on validation set if provided
        results = {}
        if X_val is not None:
            logger.info("Evaluating on validation set...")
            val_results = self.model.evaluate(X_val_transformed.toarray(), y_val)
            results['validation'] = val_results
            
            # Generate confusion matrix for validation set
            if hasattr(self.model, 'plot_confusion_matrix'):
                self.model.plot_confusion_matrix(
                    X_val_transformed.toarray(), y_val,
                    save_path=str(self.results_dir / "visualizations" / f"{self.algorithm_name}_validation_confusion_matrix.png")
                )
        
        logger.info(f"{self.algorithm_name} model training completed!")
        return results
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate the model using the algorithm class's built-in evaluation methods.
        
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
        
        # Evaluate using the algorithm class's built-in method
        test_results = self.model.evaluate(X_test_transformed.toarray(), y_test)
        
        # Generate confusion matrix
        if hasattr(self.model, 'plot_confusion_matrix'):
            self.model.plot_confusion_matrix(
                X_test_transformed.toarray(), y_test,
                save_path=str(self.results_dir / "visualizations" / f"{self.algorithm_name}_test_confusion_matrix.png")
            )
        
        # Generate additional visualizations if available
        if hasattr(self.model, 'plot_feature_importance'):
            self.model.plot_feature_importance(
                save_path=str(self.results_dir / "visualizations" / f"{self.algorithm_name}_feature_importance.png")
            )
        
        if hasattr(self.model, 'plot_learning_curve'):
            self.model.plot_learning_curve(
                X_test_transformed.toarray(), y_test,
                save_path=str(self.results_dir / "visualizations" / f"{self.algorithm_name}_learning_curve.png")
            )
        
        logger.info(f"{self.algorithm_name} model evaluation completed!")
        return test_results
    
    def save_model(self) -> None:
        """
        Save the trained model and vectorizer using the algorithm class's built-in methods.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        logger.info(f"Saving {self.algorithm_name} model...")
        
        # Save model using the algorithm class's built-in method
        if hasattr(self.model, 'save_model'):
            model_path = self.results_dir / "models" / f"{self.algorithm_name}_model.pkl"
            self.model.save_model(str(model_path))
        else:
            # Fallback to manual saving
            model_path = self.results_dir / "models" / f"{self.algorithm_name}_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
        
        # Save vectorizer
        vectorizer_path = self.results_dir / "models" / f"{self.algorithm_name}_vectorizer.pkl"
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        # Save label encoder
        encoder_path = self.results_dir / "models" / f"{self.algorithm_name}_label_encoder.pkl"
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        logger.info(f"Model, vectorizer, and label encoder saved to {self.results_dir}")
    
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
        X_train, X_val, X_test, y_train, y_val, y_test = self.load_data()
        
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
            'encoder_path': str(self.results_dir / "models" / f"{algorithm_name}_label_encoder.pkl")
        }
        
        logger.info(f"Complete pipeline finished for {algorithm_name}")
        return results


def main():
    """Main function to run the universal trainer."""
    parser = argparse.ArgumentParser(description='Universal Model Trainer for 20 Newsgroups Classification')
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
            precision = test_metrics.get('precision', 'N/A')
            recall = test_metrics.get('recall', 'N/A')
            f1_score = test_metrics.get('f1_score', 'N/A')
            
            print(f"Test Accuracy: {accuracy:.4f}" if isinstance(accuracy, (int, float)) else f"Test Accuracy: {accuracy}")
            print(f"Test Precision: {precision:.4f}" if isinstance(precision, (int, float)) else f"Test Precision: {precision}")
            print(f"Test Recall: {recall:.4f}" if isinstance(recall, (int, float)) else f"Test Recall: {recall}")
            print(f"Test F1-Score: {f1_score:.4f}" if isinstance(f1_score, (int, float)) else f"Test F1-Score: {f1_score}")
        
        print(f"\nModel saved to: {results['model_path']}")
        print(f"Vectorizer saved to: {results['vectorizer_path']}")
        print(f"Label encoder saved to: {results['encoder_path']}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        print(f"❌ Training failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
