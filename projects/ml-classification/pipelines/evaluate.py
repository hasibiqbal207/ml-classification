"""
Evaluation pipeline for machine learning models.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import json
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Add project root and repository root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(Path(__file__).parent.parent))

from utils.data_utils import DataLoader, DataPreprocessor, load_dataset
from utils.config_loader import ConfigLoader
from utils.metrics import ClassificationMetrics, ClusteringMetrics, compare_models
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

# Import new algorithms
from algorithms.classification.svm.sklearn_impl import SVMSklearn
from algorithms.classification.random_forest.sklearn_impl import RandomForestSklearn
from algorithms.classification.knn.sklearn_impl import KNNSklearn
from algorithms.classification.naive_bayes.sklearn_impl import NaiveBayesSklearn
from algorithms.classification.mlp.tensorflow_impl import MLPTensorFlow
from algorithms.classification.adaboost.sklearn_impl import AdaBoostSklearn


class EvaluationPipeline:
    """
    Evaluation pipeline for machine learning models.
    """
    
    def __init__(self, config_path="config.yaml"):
        """
        Initialize EvaluationPipeline.
        
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
        
    def load_model(self, model_path, algorithm, implementation):
        """
        Load a trained model.
        
        Args:
            model_path (str): Path to saved model
            algorithm (str): Algorithm name
            implementation (str): Implementation framework
            
        Returns:
            Loaded model
        """
        # Get model configuration
        model_config = self.config_loader.get_model_config(algorithm)
        config = model_config.get(implementation, {})
        
        # Create model instance
        model = self._create_model(algorithm, implementation, config)
        
        # Load model weights
        if implementation in ['tensorflow', 'pytorch']:
            if implementation == 'tensorflow':
                trainer = TensorFlowTrainer(model, f"{algorithm}_{implementation}")
                trainer.load_model(model_path)
                self.models[f"{algorithm}_{implementation}"] = trainer
            else:  # pytorch
                trainer = PyTorchTrainer(model, f"{algorithm}_{implementation}")
                trainer.load_model(model_path)
                self.models[f"{algorithm}_{implementation}"] = trainer
        else:
            # For sklearn models, we would need to implement loading
            # This is a simplified version
            import joblib
            model = joblib.load(model_path)
            self.models[f"{algorithm}_{implementation}"] = model
            
        print(f"Model loaded: {algorithm} ({implementation})")
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
                
        elif algorithm == 'svm':
            if implementation == 'sklearn':
                return SVMSklearn(**config)
                
        elif algorithm == 'random_forest':
            if implementation == 'sklearn':
                return RandomForestSklearn(**config)
                
        elif algorithm == 'knn':
            if implementation == 'sklearn':
                return KNNSklearn(**config)
                
        elif algorithm == 'naive_bayes':
            if implementation == 'sklearn':
                return NaiveBayesSklearn(**config)
                
        elif algorithm == 'mlp':
            if implementation == 'tensorflow':
                return MLPTensorFlow(**config)
                
        elif algorithm == 'adaboost':
            if implementation == 'sklearn':
                return AdaBoostSklearn(**config)
                
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
        
    def cross_validate(self, algorithm, implementation, cv=5):
        """
        Perform cross-validation on a model.
        
        Args:
            algorithm (str): Algorithm name
            implementation (str): Implementation framework
            cv (int): Number of cross-validation folds
            
        Returns:
            dict: Cross-validation results
        """
        if not hasattr(self, 'processed_data'):
            raise ValueError("Data must be preprocessed before cross-validation")
            
        X = self.processed_data['X_train']
        y = self.processed_data['y_train']
        
        # Get model configuration
        model_config = self.config_loader.get_model_config(algorithm)
        config = model_config.get(implementation, {})
        
        # Create model instance
        model = self._create_model(algorithm, implementation, config)
        
        # Perform cross-validation
        if implementation == 'sklearn':
            # Use sklearn's cross_val_score
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
            
            results = {
                'cv_scores': cv_scores,
                'mean_score': cv_scores.mean(),
                'std_score': cv_scores.std(),
                'algorithm': algorithm,
                'implementation': implementation
            }
        else:
            # For deep learning models, we would need to implement custom CV
            # This is a simplified version
            print(f"Cross-validation for {implementation} models not implemented yet")
            results = None
            
        print(f"Cross-validation completed for {algorithm} ({implementation})")
        return results
        
    def compare_models(self, metric='accuracy'):
        """
        Compare all evaluated models.
        
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
                    
    def generate_report(self, output_path="evaluation_report.html"):
        """
        Generate comprehensive evaluation report.
        
        Args:
            output_path (str): Path to save the report
        """
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        
        # Create PDF report
        with PdfPages(output_path.replace('.html', '.pdf')) as pdf:
            # Title page
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, 'Machine Learning Model Evaluation Report', 
                   ha='center', va='center', fontsize=20, fontweight='bold')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Model comparison
            if self.results:
                comparison_df = self.compare_models('accuracy')
                
                fig, ax = plt.subplots(figsize=(10, 6))
                comparison_df.plot(x='Model', y='Accuracy', kind='bar', ax=ax)
                ax.set_title('Model Accuracy Comparison')
                ax.set_ylabel('Accuracy')
                plt.xticks(rotation=45)
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
                
        print(f"Evaluation report saved to {output_path}")
        
    def save_results(self, filepath="evaluation_results.json"):
        """
        Save evaluation results to file.
        
        Args:
            filepath (str): Path to save results
        """
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
            
        print(f"Evaluation results saved to {filepath}")


def main():
    """
    Main function for command-line interface.
    """
    parser = argparse.ArgumentParser(description='Machine Learning Evaluation Pipeline')
    parser.add_argument('--algorithm', type=str, required=True,
                      choices=['logistic_regression', 'decision_tree', 'cnn', 'lstm', 'kmeans', 
                              'svm', 'random_forest', 'knn', 'naive_bayes', 'mlp', 'adaboost'],
                      help='Algorithm to evaluate')
    parser.add_argument('--implementation', type=str, required=True,
                      choices=['sklearn', 'tensorflow', 'pytorch'],
                      help='Implementation framework')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to saved model')
    parser.add_argument('--dataset', type=str, default='iris',
                      choices=['iris', 'wine', 'breast_cancer'],
                      help='Dataset to use')
    parser.add_argument('--config', type=str, default='config.yaml',
                      help='Configuration file path')
    parser.add_argument('--cross_validate', action='store_true',
                      help='Perform cross-validation')
    parser.add_argument('--visualize', action='store_true',
                      help='Visualize results')
    parser.add_argument('--report', action='store_true',
                      help='Generate evaluation report')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = EvaluationPipeline(args.config)
    
    # Load data
    pipeline.load_data(dataset_name=args.dataset)
    
    # Preprocess data
    pipeline.preprocess_data()
    
    # Load model
    pipeline.load_model(args.model_path, args.algorithm, args.implementation)
    
    # Evaluate model
    results = pipeline.evaluate_model(f"{args.algorithm}_{args.implementation}")
    print(f"Results: {results}")
    
    # Cross-validation
    if args.cross_validate:
        cv_results = pipeline.cross_validate(args.algorithm, args.implementation)
        if cv_results:
            print(f"Cross-validation results: {cv_results}")
            
    # Visualize results
    if args.visualize:
        pipeline.visualize_results(f"{args.algorithm}_{args.implementation}")
        
    # Generate report
    if args.report:
        pipeline.generate_report()
        
    # Save results
    pipeline.save_results()
        
    print("Evaluation pipeline execution completed!")


if __name__ == "__main__":
    main()
