"""
Main training pipeline for machine learning models.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Add project root and repository root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(Path(__file__).parent.parent))

from utils.data_utils import DataLoader, DataPreprocessor, load_dataset
from utils.config_loader import ConfigLoader, create_default_config
from utils.metrics import ClassificationMetrics, ClusteringMetrics
from utils.visualization import DataVisualizer, ModelVisualizer
from utils.trainer_tf import TensorFlowTrainer
from utils.trainer_pt import PyTorchTrainer

# Import algorithm implementations from root algorithms directory
from algorithms.classification.logistic_regression.sklearn_impl import LogisticRegressionSklearn
from algorithms.classification.logistic_regression.tensorflow_impl import LogisticRegressionTensorFlow
from algorithms.classification.logistic_regression.pytorch_impl import LogisticRegressionPyTorch
from algorithms.classification.decision_tree.sklearn_impl import DecisionTreeSklearn
from algorithms.classification.decision_tree.pytorch_impl import DecisionTreePyTorch
from algorithms.classification.cnn.tensorflow_impl import CNNTensorFlow
from algorithms.classification.cnn.pytorch_impl import CNNPyTorch
from algorithms.nlp.lstm.tensorflow_impl import LSTMTensorFlow
from algorithms.nlp.lstm.pytorch_impl import LSTMPyTorch
from algorithms.clustering.kmeans.sklearn_impl import KMeansSklearn


class MLPipeline:
    """
    Main machine learning pipeline class.
    """
    
    def __init__(self, config_path="config.yaml"):
        """
        Initialize MLPipeline.
        
        Args:
            config_path (str): Path to configuration file
        """
        self.config_loader = ConfigLoader(config_path)
        self.config_loader.load_config()
        
        self.data_loader = DataLoader()
        self.preprocessor = DataPreprocessor()
        self.metrics = ClassificationMetrics()
        self.clustering_metrics = ClusteringMetrics()
        self.visualizer = DataVisualizer()
        self.model_visualizer = ModelVisualizer()
        
        self.models = {}
        self.results = {}
        
    def load_data(self, dataset_name=None, data_path=None):
        """
        Load dataset.
        
        Args:
            dataset_name (str): Name of dataset to load
            data_path (str): Path to data file
            
        Returns:
            tuple: (X, y, feature_names, target_names)
        """
        if dataset_name:
            X, y, feature_names, target_names = load_dataset(dataset_name)
        elif data_path:
            data = self.data_loader.load_csv(data_path)
            # Assume last column is target
            X = data.iloc[:, :-1].values
            y = data.iloc[:, -1].values
            feature_names = data.columns[:-1].tolist()
            target_names = np.unique(y).tolist()
        else:
            raise ValueError("Either dataset_name or data_path must be provided")
            
        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.target_names = target_names
        
        print(f"Loaded dataset with shape: {X.shape}")
        print(f"Features: {len(feature_names)}")
        print(f"Classes: {len(target_names)}")
        
        return X, y, feature_names, target_names
        
    def preprocess_data(self, X=None, y=None):
        """
        Preprocess the data.
        
        Args:
            X (array-like): Features
            y (array-like): Labels
            
        Returns:
            dict: Preprocessed data
        """
        if X is None:
            X = self.X
        if y is None:
            y = self.y
            
        # Get preprocessing config
        preprocess_config = self.config_loader.get('data.preprocessing', {})
        
        # Preprocess data
        processed_data = self.preprocessor.preprocess_pipeline(
            X, y,
            handle_missing=preprocess_config.get('handle_missing', True),
            encode_categorical=preprocess_config.get('encode_categorical', True),
            scale_features=preprocess_config.get('scale_features', True),
            split_data=True,
            test_size=self.config_loader.get('data.test_size', 0.2)
        )
        
        self.processed_data = processed_data
        print("Data preprocessing completed")
        
        return processed_data
        
    def train_model(self, algorithm, implementation, **kwargs):
        """
        Train a machine learning model.
        
        Args:
            algorithm (str): Algorithm name
            implementation (str): Implementation framework
            **kwargs: Additional model parameters
            
        Returns:
            Trained model
        """
        if not hasattr(self, 'processed_data'):
            raise ValueError("Data must be preprocessed before training")
            
        X_train = self.processed_data['X_train']
        X_test = self.processed_data['X_test']
        y_train = self.processed_data['y_train']
        y_test = self.processed_data['y_test']
        
        # Get model configuration
        model_config = self.config_loader.get_model_config(algorithm)
        config = model_config.get(implementation, {})
        
        # Update config with kwargs
        config.update(kwargs)
        
        # Initialize model based on algorithm and implementation
        model = self._create_model(algorithm, implementation, config)
        
        # Train model
        if implementation in ['tensorflow', 'pytorch']:
            # Deep learning models
            if implementation == 'tensorflow':
                trainer = TensorFlowTrainer(model, f"{algorithm}_{implementation}")
                trainer.compile_model(**config)
                history = trainer.train(X_train, y_train, X_test, y_test, **config)
                self.models[f"{algorithm}_{implementation}"] = trainer
            else:  # pytorch
                trainer = PyTorchTrainer(model, f"{algorithm}_{implementation}")
                history = trainer.train(X_train, y_train, X_test, y_test, **config)
                self.models[f"{algorithm}_{implementation}"] = trainer
        else:
            # Traditional ML models
            model.fit(X_train, y_train)
            self.models[f"{algorithm}_{implementation}"] = model
            
        print(f"Training completed for {algorithm} ({implementation})")
        
        return model
        
    def _create_model(self, algorithm, implementation, config):
        """
        Create model instance based on algorithm and implementation.
        
        Args:
            algorithm (str): Algorithm name
            implementation (str): Implementation framework
            config (dict): Model configuration
            
        Returns:
            Model instance
        """
        if algorithm == 'logistic_regression':
            if implementation == 'sklearn':
                return LogisticRegressionSklearn(**config)
            elif implementation == 'tensorflow':
                return LogisticRegressionTensorFlow(**config)
            elif implementation == 'pytorch':
                return LogisticRegressionPyTorch(**config)
                
        elif algorithm == 'decision_tree':
            if implementation == 'sklearn':
                return DecisionTreeSklearn(**config)
            elif implementation == 'pytorch':
                return DecisionTreePyTorch(**config)
                
        elif algorithm == 'cnn':
            if implementation == 'tensorflow':
                return CNNTensorFlow(**config)
            elif implementation == 'pytorch':
                return CNNPyTorch(**config)
                
        elif algorithm == 'lstm':
            if implementation == 'tensorflow':
                return LSTMTensorFlow(**config)
            elif implementation == 'pytorch':
                return LSTMPyTorch(**config)
                
        elif algorithm == 'kmeans':
            if implementation == 'sklearn':
                return KMeansSklearn(**config)
                
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
            
    def evaluate_model(self, model_name):
        """
        Evaluate a trained model.
        
        Args:
            model_name (str): Name of the model to evaluate
            
        Returns:
            dict: Evaluation results
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
            
        model = self.models[model_name]
        X_test = self.processed_data['X_test']
        y_test = self.processed_data['y_test']
        
        # Evaluate model
        if hasattr(model, 'evaluate'):
            # Deep learning models
            results = model.evaluate(X_test, y_test)
        else:
            # Traditional ML models
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
            
            results = self.metrics.calculate_metrics(y_test, y_pred, y_proba)
            
        self.results[model_name] = results
        
        print(f"Evaluation completed for {model_name}")
        return results
        
    def compare_models(self, metric='accuracy'):
        """
        Compare all trained models.
        
        Args:
            metric (str): Metric to compare
            
        Returns:
            pd.DataFrame: Comparison results
        """
        if not self.results:
            raise ValueError("No models have been evaluated yet")
            
        comparison_data = []
        for model_name, results in self.results.items():
            comparison_data.append({
                'Model': model_name,
                metric.capitalize(): results.get(metric, 0)
            })
            
        df = pd.DataFrame(comparison_data)
        df = df.sort_values(metric.capitalize(), ascending=False)
        
        print(f"Model Comparison ({metric}):")
        print(df.to_string(index=False))
        
        return df
        
    def visualize_results(self, model_name=None):
        """
        Visualize model results.
        
        Args:
            model_name (str): Name of model to visualize (if None, visualize all)
        """
        if model_name:
            if model_name not in self.results:
                raise ValueError(f"Model {model_name} not found")
            models_to_plot = [model_name]
        else:
            models_to_plot = list(self.results.keys())
            
        for model in models_to_plot:
            if model in self.models:
                # Plot confusion matrix
                y_pred = self.models[model].predict(self.processed_data['X_test'])
                self.model_visualizer.plot_confusion_matrix(
                    self.processed_data['y_test'], y_pred,
                    title=f"Confusion Matrix - {model}"
                )
                
                # Plot ROC curve if probabilities available
                if hasattr(self.models[model], 'predict_proba'):
                    y_proba = self.models[model].predict_proba(self.processed_data['X_test'])
                    self.model_visualizer.plot_roc_curve(
                        self.processed_data['y_test'], y_proba,
                        title=f"ROC Curve - {model}"
                    )
                    
    def save_results(self, filepath="results.json"):
        """
        Save results to file.
        
        Args:
            filepath (str): Path to save results
        """
        import json
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for model_name, results in self.results.items():
            serializable_results[model_name] = {}
            for metric, value in results.items():
                if isinstance(value, np.ndarray):
                    serializable_results[model_name][metric] = value.tolist()
                else:
                    serializable_results[model_name][metric] = value
                    
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=4)
            
        print(f"Results saved to {filepath}")


def main():
    """
    Main function for command-line interface.
    """
    parser = argparse.ArgumentParser(description='Machine Learning Training Pipeline')
    parser.add_argument('--algorithm', type=str, required=True,
                      choices=['logistic_regression', 'decision_tree', 'cnn', 'lstm', 'kmeans'],
                      help='Algorithm to train')
    parser.add_argument('--implementation', type=str, required=True,
                      choices=['sklearn', 'tensorflow', 'pytorch'],
                      help='Implementation framework')
    parser.add_argument('--dataset', type=str, default='iris',
                      choices=['iris', 'wine', 'breast_cancer'],
                      help='Dataset to use')
    parser.add_argument('--config', type=str, default='config.yaml',
                      help='Configuration file path')
    parser.add_argument('--evaluate', action='store_true',
                      help='Evaluate the trained model')
    parser.add_argument('--visualize', action='store_true',
                      help='Visualize results')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = MLPipeline(args.config)
    
    # Load data
    pipeline.load_data(dataset_name=args.dataset)
    
    # Preprocess data
    pipeline.preprocess_data()
    
    # Train model
    model = pipeline.train_model(args.algorithm, args.implementation)
    
    # Evaluate model
    if args.evaluate:
        results = pipeline.evaluate_model(f"{args.algorithm}_{args.implementation}")
        print(f"Results: {results}")
        
    # Visualize results
    if args.visualize:
        pipeline.visualize_results(f"{args.algorithm}_{args.implementation}")
        
    print("Pipeline execution completed!")


if __name__ == "__main__":
    main()
