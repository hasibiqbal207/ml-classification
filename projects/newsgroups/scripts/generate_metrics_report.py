#!/usr/bin/env python3
"""
Comprehensive Metrics Report Generator for 20 Newsgroups Classification

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
from sklearn.metrics import classification_report, confusion_matrix
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
            tuple: (X_test, y_test, label_encoder)
        """
        test_path = self.data_dir / "processed" / "test.csv"
        if not test_path.exists():
            raise FileNotFoundError("Test data not found. Run preprocess_data.py first.")
        
        test_df = pd.read_csv(test_path)
        X_test = test_df['text'].values
        y_test = test_df['label'].values
        
        return X_test, y_test
    
    def load_model(self, algorithm_name: str) -> tuple:
        """
        Load a trained model and its components.
        
        Args:
            algorithm_name: Name of the algorithm
            
        Returns:
            tuple: (model, vectorizer, label_encoder)
        """
        model_path = self.results_dir / "models" / f"{algorithm_name}_model.pkl"
        vectorizer_path = self.results_dir / "models" / f"{algorithm_name}_vectorizer.pkl"
        encoder_path = self.results_dir / "models" / f"{algorithm_name}_label_encoder.pkl"
        
        if not all([model_path.exists(), vectorizer_path.exists(), encoder_path.exists()]):
            raise FileNotFoundError(f"Model files not found for {algorithm_name}")
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        
        with open(encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
        
        return model, vectorizer, label_encoder
    
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
            model, vectorizer, label_encoder = self.load_model(algorithm_name)
            
            # Load test data
            X_test, y_test = self.test_data
            
            # Transform test data
            X_test_transformed = vectorizer.transform(X_test)
            
            # Encode labels
            y_test_encoded = label_encoder.transform(y_test)
            
            # Make predictions
            y_pred = model.predict(X_test_transformed.toarray())
            
            # Calculate metrics
            accuracy = np.mean(y_pred == y_test_encoded)
            
            # Classification report
            class_report = classification_report(
                y_test_encoded, y_pred, 
                target_names=label_encoder.classes_,
                output_dict=True
            )
            
            # Confusion matrix
            cm = confusion_matrix(y_test_encoded, y_pred)
            
            # Calculate macro averages
            macro_precision = class_report['macro avg']['precision']
            macro_recall = class_report['macro avg']['recall']
            macro_f1 = class_report['macro avg']['f1-score']
            
            # Calculate weighted averages
            weighted_precision = class_report['weighted avg']['precision']
            weighted_recall = class_report['weighted avg']['recall']
            weighted_f1 = class_report['weighted avg']['f1-score']
            
            results = {
                'algorithm': algorithm_name,
                'accuracy': accuracy,
                'macro_precision': macro_precision,
                'macro_recall': macro_recall,
                'macro_f1': macro_f1,
                'weighted_precision': weighted_precision,
                'weighted_recall': weighted_recall,
                'weighted_f1': weighted_f1,
                'classification_report': class_report,
                'confusion_matrix': cm,
                'label_encoder': label_encoder
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
        macro_f1 = [successful_results[alg]['macro_f1'] for alg in algorithms]
        weighted_f1 = [successful_results[alg]['weighted_f1'] for alg in algorithms]
        
        # Create comparison chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
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
        
        bars2 = ax2.bar(x - width, macro_f1, width, label='Macro F1', alpha=0.7)
        bars3 = ax2.bar(x, weighted_f1, width, label='Weighted F1', alpha=0.7)
        
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
        
        plt.tight_layout()
        
        # Save chart
        chart_path = self.results_dir / "visualizations" / "model_comparison_charts.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Model comparison chart saved to {chart_path}")
    
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
        report.append("20 NEWSCROUPS CLASSIFICATION - DETAILED METRICS REPORT")
        report.append("="*80)
        report.append(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Summary table
        successful_results = {k: v for k, v in results.items() if 'error' not in v}
        
        if successful_results:
            report.append("MODEL PERFORMANCE SUMMARY")
            report.append("-" * 50)
            report.append(f"{'Algorithm':<20} {'Accuracy':<10} {'Macro F1':<10} {'Weighted F1':<12}")
            report.append("-" * 50)
            
            for alg_name, result in successful_results.items():
                accuracy = result['accuracy']
                macro_f1 = result['macro_f1']
                weighted_f1 = result['weighted_f1']
                report.append(f"{alg_name:<20} {accuracy:<10.4f} {macro_f1:<10.4f} {weighted_f1:<12.4f}")
            
            report.append("")
            
            # Best performing model
            best_model = max(successful_results.items(), key=lambda x: x[1]['accuracy'])
            report.append(f"BEST PERFORMING MODEL: {best_model[0].upper()}")
            report.append(f"Accuracy: {best_model[1]['accuracy']:.4f}")
            report.append(f"Macro F1-Score: {best_model[1]['macro_f1']:.4f}")
            report.append(f"Weighted F1-Score: {best_model[1]['weighted_f1']:.4f}")
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
            report.append(f"Macro Precision: {result['macro_precision']:.4f}")
            report.append(f"Macro Recall: {result['macro_recall']:.4f}")
            report.append(f"Macro F1-Score: {result['macro_f1']:.4f}")
            report.append(f"Weighted Precision: {result['weighted_precision']:.4f}")
            report.append(f"Weighted Recall: {result['weighted_recall']:.4f}")
            report.append(f"Weighted F1-Score: {result['weighted_f1']:.4f}")
            report.append("")
            
            # Per-class performance
            report.append("PER-CLASS PERFORMANCE:")
            report.append("-" * 30)
            class_report = result['classification_report']
            
            for class_name in result['label_encoder'].classes_:
                if class_name in class_report:
                    metrics = class_report[class_name]
                    report.append(f"{class_name:<25} P:{metrics['precision']:.3f} R:{metrics['recall']:.3f} F1:{metrics['f1-score']:.3f}")
            
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
                macro_f1 = result['macro_f1']
                print(f"  {alg_name}: Accuracy={accuracy:.4f}, Macro F1={macro_f1:.4f}")
        
        if failed:
            print(f"\n❌ FAILED MODELS:")
            for alg_name, result in failed.items():
                print(f"  {alg_name}: {result['error']}")
        
        print(f"\n{'='*60}")


def main():
    """Main function to run metrics generation."""
    parser = argparse.ArgumentParser(description='Generate comprehensive metrics report for 20 Newsgroups Classification')
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
