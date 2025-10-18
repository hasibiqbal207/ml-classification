"""
Comprehensive Algorithm Comparison Framework

This module provides tools for comparing multiple machine learning algorithms
across different datasets and metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import time
import warnings
warnings.filterwarnings('ignore')


class AlgorithmComparator:
    """
    Comprehensive algorithm comparison framework.
    """
    
    def __init__(self, algorithms=None, random_state=42):
        """
        Initialize AlgorithmComparator.
        
        Args:
            algorithms (dict): Dictionary of algorithm configurations
            random_state (int): Random state for reproducibility
        """
        self.random_state = random_state
        self.results = {}
        self.training_times = {}
        self.prediction_times = {}
        
        # Default algorithms configuration
        if algorithms is None:
            self.algorithms = {
                'Logistic Regression': {
                    'sklearn': {'class': 'LogisticRegressionSklearn', 'params': {'random_state': random_state}}
                },
                'SVM': {
                    'sklearn': {'class': 'SVMSklearn', 'params': {'random_state': random_state}}
                },
                'Decision Tree': {
                    'sklearn': {'class': 'DecisionTreeSklearn', 'params': {'random_state': random_state}}
                },
                'Random Forest': {
                    'sklearn': {'class': 'RandomForestSklearn', 'params': {'random_state': random_state}}
                },
                'KNN': {
                    'sklearn': {'class': 'KNNSklearn', 'params': {}}
                },
                'Naive Bayes': {
                    'sklearn': {'class': 'NaiveBayesSklearn', 'params': {'random_state': random_state}}
                },
                'AdaBoost': {
                    'sklearn': {'class': 'AdaBoostSklearn', 'params': {'random_state': random_state}}
                },
                'MLP': {
                    'tensorflow': {'class': 'MLPTensorFlow', 'params': {'random_state': random_state}}
                }
            }
        else:
            self.algorithms = algorithms
            
    def compare_algorithms(self, X_train, y_train, X_test, y_test, 
                          algorithms=None, metrics=None, cv=5):
        """
        Compare multiple algorithms on the given dataset.
        
        Args:
            X_train (array-like): Training features
            y_train (array-like): Training labels
            X_test (array-like): Test features
            y_test (array-like): Test labels
            algorithms (list): List of algorithm names to compare
            metrics (list): List of metrics to compute
            cv (int): Number of cross-validation folds
            
        Returns:
            dict: Comparison results
        """
        if algorithms is None:
            algorithms = list(self.algorithms.keys())
            
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
            
        results = {}
        
        for algo_name in algorithms:
            if algo_name not in self.algorithms:
                print(f"Warning: Algorithm {algo_name} not found in configuration")
                continue
                
            print(f"Training {algo_name}...")
            
            # Get algorithm configuration
            algo_config = self.algorithms[algo_name]
            
            # Try different implementations
            for impl_name, impl_config in algo_config.items():
                try:
                    # Import and instantiate algorithm
                    algo_class = self._get_algorithm_class(impl_config['class'])
                    model = algo_class(**impl_config['params'])
                    
                    # Measure training time
                    start_time = time.time()
                    model.fit(X_train, y_train)
                    training_time = time.time() - start_time
                    
                    # Measure prediction time
                    start_time = time.time()
                    y_pred = model.predict(X_test)
                    prediction_time = time.time() - start_time
                    
                    # Get probabilities if available
                    y_proba = None
                    if hasattr(model, 'predict_proba'):
                        y_proba = model.predict_proba(X_test)
                    
                    # Calculate metrics
                    algo_results = self._calculate_metrics(y_test, y_pred, y_proba, metrics)
                    
                    # Cross-validation
                    cv_scores = self._cross_validate(model, X_train, y_train, cv)
                    
                    # Store results
                    key = f"{algo_name}_{impl_name}"
                    results[key] = {
                        'metrics': algo_results,
                        'cv_scores': cv_scores,
                        'training_time': training_time,
                        'prediction_time': prediction_time,
                        'model': model
                    }
                    
                    self.training_times[key] = training_time
                    self.prediction_times[key] = prediction_time
                    
                    print(f"✓ {algo_name} ({impl_name}) completed")
                    
                except Exception as e:
                    print(f"✗ {algo_name} ({impl_name}) failed: {str(e)}")
                    continue
                    
        self.results = results
        return results
        
    def _get_algorithm_class(self, class_name):
        """
        Dynamically import algorithm class.
        
        Args:
            class_name (str): Name of the algorithm class
            
        Returns:
            class: Algorithm class
        """
        # Import mapping
        import_mapping = {
            'LogisticRegressionSklearn': 'algorithms.classification.logistic_regression.sklearn_impl',
            'SVMSklearn': 'algorithms.classification.svm.sklearn_impl',
            'DecisionTreeSklearn': 'algorithms.classification.decision_tree.sklearn_impl',
            'RandomForestSklearn': 'algorithms.classification.random_forest.sklearn_impl',
            'KNNSklearn': 'algorithms.classification.knn.sklearn_impl',
            'NaiveBayesSklearn': 'algorithms.classification.naive_bayes.sklearn_impl',
            'AdaBoostSklearn': 'algorithms.classification.adaboost.sklearn_impl',
            'MLPTensorFlow': 'algorithms.classification.mlp.tensorflow_impl'
        }
        
        if class_name not in import_mapping:
            raise ValueError(f"Unknown algorithm class: {class_name}")
            
        module_path = import_mapping[class_name]
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)
        
    def _calculate_metrics(self, y_true, y_pred, y_proba, metrics):
        """
        Calculate specified metrics.
        
        Args:
            y_true (array-like): True labels
            y_pred (array-like): Predicted labels
            y_proba (array-like): Predicted probabilities
            metrics (list): List of metrics to calculate
            
        Returns:
            dict: Calculated metrics
        """
        results = {}
        
        for metric in metrics:
            try:
                if metric == 'accuracy':
                    results[metric] = accuracy_score(y_true, y_pred)
                elif metric == 'precision':
                    results[metric] = precision_score(y_true, y_pred, average='weighted')
                elif metric == 'recall':
                    results[metric] = recall_score(y_true, y_pred, average='weighted')
                elif metric == 'f1_score':
                    results[metric] = f1_score(y_true, y_pred, average='weighted')
                elif metric == 'roc_auc':
                    if y_proba is not None and len(np.unique(y_true)) == 2:
                        results[metric] = roc_auc_score(y_true, y_proba[:, 1])
                    else:
                        results[metric] = np.nan
                else:
                    results[metric] = np.nan
            except Exception as e:
                print(f"Warning: Could not calculate {metric}: {str(e)}")
                results[metric] = np.nan
                
        return results
        
    def _cross_validate(self, model, X, y, cv):
        """
        Perform cross-validation.
        
        Args:
            model: Trained model
            X (array-like): Features
            y (array-like): Labels
            cv (int): Number of folds
            
        Returns:
            dict: Cross-validation results
        """
        try:
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
            return {
                'mean': cv_scores.mean(),
                'std': cv_scores.std(),
                'scores': cv_scores
            }
        except Exception as e:
            print(f"Warning: Cross-validation failed: {str(e)}")
            return {'mean': np.nan, 'std': np.nan, 'scores': []}
            
    def create_comparison_table(self, metric='accuracy'):
        """
        Create a comparison table for a specific metric.
        
        Args:
            metric (str): Metric to compare
            
        Returns:
            pd.DataFrame: Comparison table
        """
        if not self.results:
            raise ValueError("No results available. Run compare_algorithms first.")
            
        data = []
        for algo_name, results in self.results.items():
            row = {
                'Algorithm': algo_name,
                f'{metric.capitalize()}': results['metrics'].get(metric, np.nan),
                'CV Mean': results['cv_scores']['mean'],
                'CV Std': results['cv_scores']['std'],
                'Training Time (s)': results['training_time'],
                'Prediction Time (s)': results['prediction_time']
            }
            data.append(row)
            
        df = pd.DataFrame(data)
        df = df.sort_values(f'{metric.capitalize()}', ascending=False)
        
        return df
        
    def plot_comparison(self, metric='accuracy', figsize=(12, 8)):
        """
        Plot algorithm comparison.
        
        Args:
            metric (str): Metric to plot
            figsize (tuple): Figure size
        """
        if not self.results:
            raise ValueError("No results available. Run compare_algorithms first.")
            
        # Prepare data
        algorithms = []
        scores = []
        cv_means = []
        cv_stds = []
        
        for algo_name, results in self.results.items():
            algorithms.append(algo_name)
            scores.append(results['metrics'].get(metric, np.nan))
            cv_means.append(results['cv_scores']['mean'])
            cv_stds.append(results['cv_scores']['std'])
            
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Test scores
        bars1 = ax1.bar(range(len(algorithms)), scores)
        ax1.set_xlabel('Algorithm')
        ax1.set_ylabel(f'{metric.capitalize()}')
        ax1.set_title(f'{metric.capitalize()} Comparison (Test Set)')
        ax1.set_xticks(range(len(algorithms)))
        ax1.set_xticklabels(algorithms, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, score in zip(bars1, scores):
            if not np.isnan(score):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{score:.3f}', ha='center', va='bottom')
        
        # CV scores with error bars
        bars2 = ax2.bar(range(len(algorithms)), cv_means, yerr=cv_stds, capsize=5)
        ax2.set_xlabel('Algorithm')
        ax2.set_ylabel(f'{metric.capitalize()}')
        ax2.set_title(f'{metric.capitalize()} Comparison (Cross-Validation)')
        ax2.set_xticks(range(len(algorithms)))
        ax2.set_xticklabels(algorithms, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, mean, std in zip(bars2, cv_means, cv_stds):
            if not np.isnan(mean):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
                        f'{mean:.3f}±{std:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
    def plot_training_times(self, figsize=(10, 6)):
        """
        Plot training times comparison.
        
        Args:
            figsize (tuple): Figure size
        """
        if not self.training_times:
            raise ValueError("No training times available. Run compare_algorithms first.")
            
        algorithms = list(self.training_times.keys())
        times = list(self.training_times.values())
        
        plt.figure(figsize=figsize)
        bars = plt.bar(range(len(algorithms)), times)
        plt.xlabel('Algorithm')
        plt.ylabel('Training Time (seconds)')
        plt.title('Training Time Comparison')
        plt.xticks(range(len(algorithms)), algorithms, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, time in zip(bars, times):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{time:.2f}s', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
    def plot_prediction_times(self, figsize=(10, 6)):
        """
        Plot prediction times comparison.
        
        Args:
            figsize (tuple): Figure size
        """
        if not self.prediction_times:
            raise ValueError("No prediction times available. Run compare_algorithms first.")
            
        algorithms = list(self.prediction_times.keys())
        times = list(self.prediction_times.values())
        
        plt.figure(figsize=figsize)
        bars = plt.bar(range(len(algorithms)), times)
        plt.xlabel('Algorithm')
        plt.ylabel('Prediction Time (seconds)')
        plt.title('Prediction Time Comparison')
        plt.xticks(range(len(algorithms)), algorithms, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, time in zip(bars, times):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{time:.4f}s', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
    def plot_confusion_matrices(self, X_test, y_test, figsize=(15, 10)):
        """
        Plot confusion matrices for all algorithms.
        
        Args:
            X_test (array-like): Test features
            y_test (array-like): Test labels
            figsize (tuple): Figure size
        """
        if not self.results:
            raise ValueError("No results available. Run compare_algorithms first.")
            
        n_algorithms = len(self.results)
        n_cols = 3
        n_rows = (n_algorithms + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
            
        for i, (algo_name, results) in enumerate(self.results.items()):
            row = i // n_cols
            col = i % n_cols
            
            if n_rows > 1 and n_cols > 1:
                ax = axes[row, col]
            elif n_rows > 1:
                ax = axes[row]
            elif n_cols > 1:
                ax = axes[col]
            else:
                ax = axes
                
            # Get predictions
            model = results['model']
            y_pred = model.predict(X_test)
            
            # Plot confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title(f'{algo_name}')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            
        # Hide empty subplots
        for i in range(n_algorithms, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            if n_rows > 1 and n_cols > 1:
                axes[row, col].set_visible(False)
            elif n_rows > 1:
                axes[row].set_visible(False)
            elif n_cols > 1:
                axes[col].set_visible(False)
                
        plt.tight_layout()
        plt.show()
        
    def get_best_algorithm(self, metric='accuracy'):
        """
        Get the best performing algorithm for a given metric.
        
        Args:
            metric (str): Metric to optimize
            
        Returns:
            tuple: (algorithm_name, score)
        """
        if not self.results:
            raise ValueError("No results available. Run compare_algorithms first.")
            
        best_score = -np.inf
        best_algorithm = None
        
        for algo_name, results in self.results.items():
            score = results['metrics'].get(metric, np.nan)
            if not np.isnan(score) and score > best_score:
                best_score = score
                best_algorithm = algo_name
                
        return best_algorithm, best_score
        
    def save_results(self, filepath='algorithm_comparison_results.csv'):
        """
        Save comparison results to CSV.
        
        Args:
            filepath (str): Path to save results
        """
        if not self.results:
            raise ValueError("No results available. Run compare_algorithms first.")
            
        # Create comprehensive results table
        data = []
        for algo_name, results in self.results.items():
            row = {
                'Algorithm': algo_name,
                'Accuracy': results['metrics'].get('accuracy', np.nan),
                'Precision': results['metrics'].get('precision', np.nan),
                'Recall': results['metrics'].get('recall', np.nan),
                'F1_Score': results['metrics'].get('f1_score', np.nan),
                'ROC_AUC': results['metrics'].get('roc_auc', np.nan),
                'CV_Mean': results['cv_scores']['mean'],
                'CV_Std': results['cv_scores']['std'],
                'Training_Time': results['training_time'],
                'Prediction_Time': results['prediction_time']
            }
            data.append(row)
            
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        print(f"Results saved to {filepath}")
        
    def print_summary(self):
        """
        Print a summary of the comparison results.
        """
        if not self.results:
            print("No results available. Run compare_algorithms first.")
            return
            
        print("=" * 80)
        print("ALGORITHM COMPARISON SUMMARY")
        print("=" * 80)
        
        # Best algorithms for each metric
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        for metric in metrics:
            best_algo, best_score = self.get_best_algorithm(metric)
            if best_algo:
                print(f"Best {metric.capitalize()}: {best_algo} ({best_score:.4f})")
                
        print("\n" + "-" * 80)
        
        # Performance table
        df = self.create_comparison_table('accuracy')
        print("PERFORMANCE COMPARISON:")
        print(df.to_string(index=False, float_format='%.4f'))
        
        print("\n" + "-" * 80)
        
        # Timing summary
        print("TIMING SUMMARY:")
        print(f"Fastest Training: {min(self.training_times, key=self.training_times.get)} "
              f"({min(self.training_times.values()):.4f}s)")
        print(f"Fastest Prediction: {min(self.prediction_times, key=self.prediction_times.get)} "
              f"({min(self.prediction_times.values()):.4f}s)")
        
        print("=" * 80)
