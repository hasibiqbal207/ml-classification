#!/usr/bin/env python3
"""
Comprehensive Metrics Report Generator for GoEmotions Multilabel Classification

This script generates detailed performance reports for all trained models.
"""

import os
import sys
import pickle
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss, jaccard_score
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MetricsReportGenerator:
    """
    Generator for comprehensive metrics reports.
    """
    
    def __init__(self, data_dir: str = None, results_dir: str = None):
        """
        Initialize the metrics generator.
        
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
        
        # Create reports directory
        (self.results_dir / "reports").mkdir(exist_ok=True)
        
        # Available algorithms
        self.available_algorithms = [
            'naive_bayes', 'logistic_regression', 'random_forest', 
            'svm', 'knn', 'adaboost', 'decision_tree'
        ]
        
        # Load test data
        self.test_data = self._load_test_data()
        
    def _load_test_data(self) -> tuple:
        """
        Load test data for evaluation.
        
        Returns:
            tuple: (X_test, y_test, emotion_categories)
        """
        test_texts_path = self.data_dir / "processed" / "test_texts.csv"
        test_labels_path = self.data_dir / "processed" / "test_labels.csv"
        categories_path = self.data_dir / "processed" / "emotion_categories.pkl"
        
        if not all([test_texts_path.exists(), test_labels_path.exists(), categories_path.exists()]):
            raise FileNotFoundError("Test data not found. Run preprocess_data.py first.")
        
        X_test = pd.read_csv(test_texts_path)['text'].values
        y_test = pd.read_csv(test_labels_path).values
        
        with open(categories_path, 'rb') as f:
            emotion_categories = pickle.load(f)
        
        return X_test, y_test, emotion_categories
    
    def load_model(self, algorithm_name: str) -> tuple:
        """
        Load a trained model and its components.
        
        Args:
            algorithm_name: Name of the algorithm
            
        Returns:
            tuple: (model, vectorizer, emotion_categories)
        """
        model_path = self.results_dir / "models" / f"{algorithm_name}_model.pkl"
        vectorizer_path = self.results_dir / "models" / f"{algorithm_name}_vectorizer.pkl"
        categories_path = self.results_dir / "models" / f"{algorithm_name}_emotion_categories.pkl"
        
        if not all([model_path.exists(), vectorizer_path.exists(), categories_path.exists()]):
            raise FileNotFoundError(f"Model files not found for {algorithm_name}")
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        
        with open(categories_path, 'rb') as f:
            emotion_categories = pickle.load(f)
        
        return model, vectorizer, emotion_categories
    
    def evaluate_model(self, algorithm_name: str) -> Dict[str, Any]:
        """
        Evaluate a single model.
        
        Args:
            algorithm_name: Name of the algorithm
            
        Returns:
            dict: Evaluation results
        """
        logger.info(f"Evaluating {algorithm_name} model...")
        
        try:
            # Load model
            model, vectorizer, emotion_categories = self.load_model(algorithm_name)
            
            # Load test data
            X_test, y_test, _ = self.test_data
            
            # Transform test data
            X_test_transformed = vectorizer.transform(X_test)
            
            # Make predictions
            y_pred = model.predict(X_test_transformed.toarray())
            
            # Calculate multilabel metrics
            accuracy = accuracy_score(y_test, y_pred)
            hamming_loss_val = hamming_loss(y_test, y_pred)
            jaccard_score_val = jaccard_score(y_test, y_pred, average='samples')
            
            # Calculate precision, recall, f1 for different averaging methods
            precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
            precision_micro = precision_score(y_test, y_pred, average='micro', zero_division=0)
            recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
            recall_micro = recall_score(y_test, y_pred, average='micro', zero_division=0)
            f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
            f1_micro = f1_score(y_test, y_pred, average='micro', zero_division=0)
            
            # Calculate per-emotion metrics
            per_emotion_metrics = {}
            for i, emotion in enumerate(emotion_categories):
                per_emotion_metrics[emotion] = {
                    'precision': precision_score(y_test[:, i], y_pred[:, i], zero_division=0),
                    'recall': recall_score(y_test[:, i], y_pred[:, i], zero_division=0),
                    'f1': f1_score(y_test[:, i], y_pred[:, i], zero_division=0),
                    'support': y_test[:, i].sum()
                }
            
            results = {
                'algorithm': algorithm_name,
                'accuracy': accuracy,
                'hamming_loss': hamming_loss_val,
                'jaccard_score': jaccard_score_val,
                'precision_macro': precision_macro,
                'precision_micro': precision_micro,
                'recall_macro': recall_macro,
                'recall_micro': recall_micro,
                'f1_macro': f1_macro,
                'f1_micro': f1_micro,
                'per_emotion': per_emotion_metrics,
                'emotion_categories': emotion_categories
            }
            
            logger.info(f"✅ {algorithm_name} evaluation completed")
            return results
            
        except Exception as e:
            logger.error(f"❌ {algorithm_name} evaluation failed: {e}")
            return {
                'algorithm': algorithm_name,
                'error': str(e),
                'status': 'failed'
            }
    
    def generate_model_comparison_chart(self, results: Dict[str, Any]) -> None:
        """
        Generate model comparison visualization.
        
        Args:
            results: Results from all models
        """
        logger.info("Generating model comparison chart...")
        
        # Filter successful results
        successful_results = {k: v for k, v in results.items() if 'error' not in v}
        
        if not successful_results:
            logger.warning("No successful results to plot")
            return
        
        # Extract metrics
        algorithms = list(successful_results.keys())
        accuracy = [successful_results[alg]['accuracy'] for alg in algorithms]
        f1_macro = [successful_results[alg]['f1_macro'] for alg in algorithms]
        f1_micro = [successful_results[alg]['f1_micro'] for alg in algorithms]
        hamming_loss = [successful_results[alg]['hamming_loss'] for alg in algorithms]
        
        # Create comparison chart
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Accuracy comparison
        bars1 = ax1.bar(algorithms, accuracy, color='skyblue', alpha=0.7)
        ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, acc in zip(bars1, accuracy):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # F1-Score comparison
        x = np.arange(len(algorithms))
        width = 0.25
        
        bars2 = ax2.bar(x - width, f1_macro, width, label='F1-Macro', alpha=0.7)
        bars3 = ax2.bar(x, f1_micro, width, label='F1-Micro', alpha=0.7)
        
        ax2.set_title('F1-Score Comparison', fontsize=14, fontweight='bold')
        ax2.set_ylabel('F1-Score', fontsize=12)
        ax2.set_xticks(x)
        ax2.set_xticklabels(algorithms, rotation=45)
        ax2.set_ylim(0, 1)
        ax2.legend()
        
        # Add value labels
        for bars in [bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Hamming Loss comparison (lower is better)
        bars4 = ax3.bar(algorithms, hamming_loss, color='lightcoral', alpha=0.7)
        ax3.set_title('Hamming Loss Comparison (Lower is Better)', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Hamming Loss', fontsize=12)
        ax3.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, loss in zip(bars4, hamming_loss):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{loss:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Jaccard Score comparison
        jaccard_scores = [successful_results[alg]['jaccard_score'] for alg in algorithms]
        bars5 = ax4.bar(algorithms, jaccard_scores, color='lightgreen', alpha=0.7)
        ax4.set_title('Jaccard Score Comparison', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Jaccard Score', fontsize=12)
        ax4.set_ylim(0, 1)
        ax4.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, score in zip(bars5, jaccard_scores):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save chart
        chart_path = self.results_dir / "visualizations" / "model_comparison_charts.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Model comparison chart saved to {chart_path}")
    
    def generate_emotion_performance_chart(self, results: Dict[str, Any]) -> None:
        """
        Generate per-emotion performance visualization.
        
        Args:
            results: Results from all models
        """
        logger.info("Generating per-emotion performance chart...")
        
        # Filter successful results
        successful_results = {k: v for k, v in results.items() if 'error' not in v}
        
        if not successful_results:
            logger.warning("No successful results to plot")
            return
        
        # Get emotion categories from first successful result
        emotion_categories = list(successful_results[list(successful_results.keys())[0]]['emotion_categories'])
        
        # Create per-emotion F1 scores matrix
        f1_scores = np.zeros((len(emotion_categories), len(successful_results)))
        algorithm_names = list(successful_results.keys())
        
        for i, emotion in enumerate(emotion_categories):
            for j, alg in enumerate(algorithm_names):
                f1_scores[i, j] = successful_results[alg]['per_emotion'][emotion]['f1']
        
        # Create heatmap
        plt.figure(figsize=(12, 16))
        sns.heatmap(f1_scores, 
                   xticklabels=algorithm_names,
                   yticklabels=emotion_categories,
                   annot=True, 
                   fmt='.3f',
                   cmap='YlOrRd',
                   cbar_kws={'label': 'F1-Score'})
        
        plt.title('Per-Emotion F1-Score Performance', fontsize=16, fontweight='bold')
        plt.xlabel('Algorithms', fontsize=12)
        plt.ylabel('Emotions', fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save chart
        chart_path = self.results_dir / "visualizations" / "emotion_performance_heatmap.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Emotion performance chart saved to {chart_path}")
    
    def generate_detailed_report(self, results: Dict[str, Any]) -> str:
        """
        Generate detailed text report.
        
        Args:
            results: Results from all models
            
        Returns:
            str: Detailed report text
        """
        logger.info("Generating detailed metrics report...")
        
        report = []
        report.append("="*80)
        report.append("GOEMOTIONS MULTILABEL CLASSIFICATION - DETAILED METRICS REPORT")
        report.append("="*80)
        report.append(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Summary table
        successful_results = {k: v for k, v in results.items() if 'error' not in v}
        
        if successful_results:
            report.append("MODEL PERFORMANCE SUMMARY")
            report.append("-" * 70)
            report.append(f"{'Algorithm':<20} {'Accuracy':<10} {'F1-Macro':<10} {'F1-Micro':<10} {'Hamming Loss':<12}")
            report.append("-" * 70)
            
            for alg_name, result in successful_results.items():
                accuracy = result['accuracy']
                f1_macro = result['f1_macro']
                f1_micro = result['f1_micro']
                hamming_loss = result['hamming_loss']
                report.append(f"{alg_name:<20} {accuracy:<10.4f} {f1_macro:<10.4f} {f1_micro:<10.4f} {hamming_loss:<12.4f}")
            
            report.append("")
            
            # Best performing model
            best_model = max(successful_results.items(), key=lambda x: x[1]['f1_macro'])
            report.append(f"BEST PERFORMING MODEL: {best_model[0].upper()}")
            report.append(f"Accuracy: {best_model[1]['accuracy']:.4f}")
            report.append(f"F1-Macro: {best_model[1]['f1_macro']:.4f}")
            report.append(f"F1-Micro: {best_model[1]['f1_micro']:.4f}")
            report.append(f"Hamming Loss: {best_model[1]['hamming_loss']:.4f}")
            report.append("")
        
        # Individual model details
        for alg_name, result in results.items():
            if 'error' in result:
                report.append(f"{alg_name.upper()} - FAILED")
                report.append(f"Error: {result['error']}")
                report.append("")
                continue
            
            report.append(f"{alg_name.upper()} - DETAILED RESULTS")
            report.append("-" * 50)
            report.append(f"Accuracy: {result['accuracy']:.4f}")
            report.append(f"Hamming Loss: {result['hamming_loss']:.4f}")
            report.append(f"Jaccard Score: {result['jaccard_score']:.4f}")
            report.append(f"Precision (Macro): {result['precision_macro']:.4f}")
            report.append(f"Precision (Micro): {result['precision_micro']:.4f}")
            report.append(f"Recall (Macro): {result['recall_macro']:.4f}")
            report.append(f"Recall (Micro): {result['recall_micro']:.4f}")
            report.append(f"F1-Score (Macro): {result['f1_macro']:.4f}")
            report.append(f"F1-Score (Micro): {result['f1_micro']:.4f}")
            report.append("")
            
            # Per-emotion performance
            report.append("PER-EMOTION PERFORMANCE:")
            report.append("-" * 40)
            per_emotion = result['per_emotion']
            
            # Sort emotions by F1-score
            sorted_emotions = sorted(per_emotion.items(), key=lambda x: x[1]['f1'], reverse=True)
            
            for emotion, metrics in sorted_emotions:
                f1 = metrics['f1']
                precision = metrics['precision']
                recall = metrics['recall']
                support = metrics['support']
                report.append(f"{emotion:<20} F1:{f1:.3f} P:{precision:.3f} R:{recall:.3f} S:{support}")
            
            report.append("")
        
        # Failed models
        failed_results = {k: v for k, v in results.items() if 'error' in v}
        if failed_results:
            report.append("FAILED MODELS")
            report.append("-" * 20)
            for alg_name, result in failed_results.items():
                report.append(f"{alg_name}: {result['error']}")
            report.append("")
        
        report.append("="*80)
        
        return "\n".join(report)
    
    def generate_all_reports(self, specific_model: str = None) -> Dict[str, Any]:
        """
        Generate reports for all models or a specific model.
        
        Args:
            specific_model: Specific model to generate report for (optional)
            
        Returns:
            dict: Results from all evaluations
        """
        logger.info("Starting comprehensive metrics generation...")
        
        # Determine which models to evaluate
        if specific_model:
            if specific_model not in self.available_algorithms:
                raise ValueError(f"Model '{specific_model}' not available")
            models_to_evaluate = [specific_model]
        else:
            models_to_evaluate = self.available_algorithms
        
        # Evaluate all models
        results = {}
        for algorithm in models_to_evaluate:
            results[algorithm] = self.evaluate_model(algorithm)
        
        # Generate visualizations
        if not specific_model:  # Only generate comparison charts for all models
            self.generate_model_comparison_chart(results)
            self.generate_emotion_performance_chart(results)
        
        # Generate detailed report
        report_text = self.generate_detailed_report(results)
        
        # Save report
        report_path = self.results_dir / "reports" / "detailed_metrics_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        logger.info(f"Detailed report saved to {report_path}")
        
        # Print summary
        self.print_summary(results)
        
        return results
    
    def print_summary(self, results: Dict[str, Any]) -> None:
        """
        Print evaluation summary.
        
        Args:
            results: Results from all evaluations
        """
        print(f"\n{'='*60}")
        print("METRICS GENERATION SUMMARY")
        print(f"{'='*60}")
        
        successful = {k: v for k, v in results.items() if 'error' not in v}
        failed = {k: v for k, v in results.items() if 'error' in v}
        
        print(f"Models evaluated: {len(results)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")
        
        if successful:
            print(f"\n✅ SUCCESSFUL MODELS:")
            for alg_name, result in successful.items():
                accuracy = result['accuracy']
                f1_macro = result['f1_macro']
                hamming_loss = result['hamming_loss']
                print(f"  {alg_name}: Accuracy={accuracy:.4f}, F1-Macro={f1_macro:.4f}, Hamming Loss={hamming_loss:.4f}")
        
        if failed:
            print(f"\n❌ FAILED MODELS:")
            for alg_name, result in failed.items():
                print(f"  {alg_name}: {result['error']}")
        
        print(f"\n{'='*60}")


def main():
    """Main function to run metrics generation."""
    parser = argparse.ArgumentParser(description='Generate comprehensive metrics report for GoEmotions Multilabel Classification')
    parser.add_argument('--model', help='Specific model to generate report for')
    parser.add_argument('--data-dir', help='Path to data directory')
    parser.add_argument('--results-dir', help='Path to results directory')
    
    args = parser.parse_args()
    
    # Create metrics generator
    generator = MetricsReportGenerator(args.data_dir, args.results_dir)
    
    # Generate reports
    try:
        results = generator.generate_all_reports(args.model)
        
        print(f"\n🎉 Metrics generation completed!")
        print(f"Check the results/reports directory for detailed reports.")
        
    except Exception as e:
        logger.error(f"Metrics generation failed: {e}")
        print(f"❌ Metrics generation failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
