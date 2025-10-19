#!/usr/bin/env python3
"""
Metrics Generation Script for SMS Spam Binary Classification

This script generates comprehensive metrics reports from confusion matrices
and model performance data for all trained models.
"""

import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))


class MetricsGenerator:
    """Generate comprehensive metrics reports from trained models."""
    
    def __init__(self, results_dir: str = None, data_dir: str = None):
        """
        Initialize the metrics generator.
        
        Args:
            results_dir: Path to results directory
            data_dir: Path to data directory
        """
        if results_dir is None:
            self.results_dir = Path(__file__).parent.parent / "results"
        else:
            self.results_dir = Path(results_dir)
            
        if data_dir is None:
            self.data_dir = Path(__file__).parent.parent / "data"
        else:
            self.data_dir = Path(data_dir)
            
        # Available models
        self.models = ['naive_bayes', 'logistic_regression', 'random_forest']
        
    def load_test_data(self):
        """Load test data for evaluation."""
        test_path = self.data_dir / "processed" / "test.csv"
        self.test_data = pd.read_csv(test_path)
        self.y_test = self.test_data['label'].map({'ham': 0, 'spam': 1})
        print(f"Loaded test data: {len(self.test_data)} samples")
        
    def load_model_and_predict(self, model_name):
        """
        Load a trained model and make predictions on test data.
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            tuple: (y_pred, y_proba, model, vectorizer)
        """
        print(f"\nLoading {model_name} model...")
        
        # Load model and vectorizer
        model_path = self.results_dir / f"{model_name}_model.pkl"
        vectorizer_path = self.results_dir / f"{model_name}_vectorizer.pkl"
        
        if not model_path.exists() or not vectorizer_path.exists():
            print(f"Model files not found for {model_name}")
            return None, None, None, None
            
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
            
        # Transform test data
        X_test = vectorizer.transform(self.test_data['text'])
        
        # Make predictions
        y_pred = model.predict(X_test.toarray())
        y_proba = model.predict_proba(X_test.toarray())
        
        print(f"Successfully loaded {model_name} model")
        return y_pred, y_proba, model, vectorizer
        
    def calculate_metrics(self, y_true, y_pred, y_proba):
        """
        Calculate comprehensive metrics from predictions.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities
            
        Returns:
            dict: Dictionary of calculated metrics
        """
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_proba[:, 1])
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Additional metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # Precision-Recall metrics
        precision_ham = tn / (tn + fn) if (tn + fn) > 0 else 0
        precision_spam = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'specificity': specificity,
            'sensitivity': sensitivity,
            'false_positive_rate': false_positive_rate,
            'false_negative_rate': false_negative_rate,
            'precision_ham': precision_ham,
            'precision_spam': precision_spam,
            'confusion_matrix': cm,
            'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
        }
        
    def generate_model_report(self, model_name):
        """
        Generate detailed report for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            dict: Model metrics and predictions
        """
        print(f"\n{'='*60}")
        print(f"GENERATING REPORT FOR {model_name.upper()}")
        print(f"{'='*60}")
        
        # Load model and make predictions
        y_pred, y_proba, model, vectorizer = self.load_model_and_predict(model_name)
        
        if y_pred is None:
            return None
            
        # Calculate metrics
        metrics = self.calculate_metrics(self.y_test, y_pred, y_proba)
        
        # Print detailed report
        print(f"\n{model_name.upper()} PERFORMANCE METRICS:")
        print(f"{'='*50}")
        print(f"Accuracy:           {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"Precision:           {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
        print(f"Recall (Sensitivity): {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
        print(f"F1-Score:            {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
        print(f"ROC-AUC:             {metrics['roc_auc']:.4f} ({metrics['roc_auc']*100:.2f}%)")
        print(f"Specificity:         {metrics['specificity']:.4f} ({metrics['specificity']*100:.2f}%)")
        print(f"False Positive Rate: {metrics['false_positive_rate']:.4f} ({metrics['false_positive_rate']*100:.2f}%)")
        print(f"False Negative Rate: {metrics['false_negative_rate']:.4f} ({metrics['false_negative_rate']*100:.2f}%)")
        
        print(f"\nCONFUSION MATRIX:")
        print(f"{'='*30}")
        print(f"True Negatives (TN):  {metrics['tn']:4d}")
        print(f"False Positives (FP): {metrics['fp']:4d}")
        print(f"False Negatives (FN): {metrics['fn']:4d}")
        print(f"True Positives (TP):  {metrics['tp']:4d}")
        
        print(f"\nCLASS-SPECIFIC METRICS:")
        print(f"{'='*30}")
        print(f"Ham Precision:       {metrics['precision_ham']:.4f} ({metrics['precision_ham']*100:.2f}%)")
        print(f"Spam Precision:      {metrics['precision_spam']:.4f} ({metrics['precision_spam']*100:.2f}%)")
        
        # Business impact analysis
        print(f"\nBUSINESS IMPACT ANALYSIS:")
        print(f"{'='*30}")
        print(f"Spam Messages Correctly Identified: {metrics['tp']}/{metrics['tp'] + metrics['fn']} ({metrics['recall']*100:.1f}%)")
        print(f"Ham Messages Correctly Identified:  {metrics['tn']}/{metrics['tn'] + metrics['fp']} ({metrics['specificity']*100:.1f}%)")
        print(f"Spam Messages Missed:               {metrics['fn']} ({metrics['false_negative_rate']*100:.1f}%)")
        print(f"Ham Messages Misclassified:         {metrics['fp']} ({metrics['false_positive_rate']*100:.1f}%)")
        
        return {
            'model_name': model_name,
            'metrics': metrics,
            'predictions': y_pred,
            'probabilities': y_proba,
            'model': model,
            'vectorizer': vectorizer
        }
        
    def generate_comparison_report(self, model_reports):
        """
        Generate comparison report across all models.
        
        Args:
            model_reports: List of model reports
        """
        print(f"\n{'='*80}")
        print("MODEL COMPARISON REPORT")
        print(f"{'='*80}")
        
        # Create comparison DataFrame
        comparison_data = []
        for report in model_reports:
            if report is not None:
                metrics = report['metrics']
                comparison_data.append({
                    'Model': report['model_name'].replace('_', ' ').title(),
                    'Accuracy': f"{metrics['accuracy']:.4f}",
                    'Precision': f"{metrics['precision']:.4f}",
                    'Recall': f"{metrics['recall']:.4f}",
                    'F1-Score': f"{metrics['f1_score']:.4f}",
                    'ROC-AUC': f"{metrics['roc_auc']:.4f}",
                    'Specificity': f"{metrics['specificity']:.4f}",
                    'FPR': f"{metrics['false_positive_rate']:.4f}",
                    'FNR': f"{metrics['false_negative_rate']:.4f}"
                })
        
        df_comparison = pd.DataFrame(comparison_data)
        print(df_comparison.to_string(index=False))
        
        # Find best performing model for each metric
        print(f"\n{'='*50}")
        print("BEST PERFORMING MODELS:")
        print(f"{'='*50}")
        
        best_models = {}
        for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
            best_score = 0
            best_model = None
            for report in model_reports:
                if report is not None:
                    score = report['metrics'][metric]
                    if score > best_score:
                        best_score = score
                        best_model = report['model_name']
            best_models[metric] = (best_model, best_score)
            
        for metric, (model, score) in best_models.items():
            print(f"{metric.replace('_', ' ').title():15}: {model.replace('_', ' ').title()} ({score:.4f})")
            
        return df_comparison
        
    def plot_comparison_charts(self, model_reports):
        """Plot comparison charts for all models."""
        print(f"\nGenerating comparison charts...")
        
        # Extract metrics for plotting
        models = []
        metrics_data = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': [],
            'roc_auc': []
        }
        
        for report in model_reports:
            if report is not None:
                models.append(report['model_name'].replace('_', ' ').title())
                for metric in metrics_data.keys():
                    metrics_data[metric].append(report['metrics'][metric])
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        metrics_list = list(metrics_data.keys())
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for i, metric in enumerate(metrics_list):
            row = i // 3
            col = i % 3
            
            bars = axes[row, col].bar(models, metrics_data[metric], color=colors[:len(models)])
            axes[row, col].set_title(f'{metric.replace("_", " ").title()}', fontweight='bold')
            axes[row, col].set_ylabel('Score')
            axes[row, col].set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars, metrics_data[metric]):
                axes[row, col].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                  f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # Rotate x-axis labels
            axes[row, col].tick_params(axis='x', rotation=45)
        
        # Remove empty subplot
        axes[1, 2].remove()
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.results_dir / "model_comparison_charts.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Comparison charts saved to: {plot_path}")
        
    def save_detailed_report(self, model_reports, comparison_df):
        """Save detailed report to file."""
        report_path = self.results_dir / "detailed_metrics_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("SMS SPAM DETECTION - DETAILED METRICS REPORT\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Test Dataset Size: {len(self.test_data)} samples\n")
            f.write(f"Ham Messages: {(self.y_test == 0).sum()} ({(self.y_test == 0).mean()*100:.1f}%)\n")
            f.write(f"Spam Messages: {(self.y_test == 1).sum()} ({(self.y_test == 1).mean()*100:.1f}%)\n\n")
            
            # Model-specific reports
            for report in model_reports:
                if report is not None:
                    f.write(f"\n{report['model_name'].upper()} MODEL REPORT\n")
                    f.write("-" * 40 + "\n")
                    metrics = report['metrics']
                    
                    f.write(f"Accuracy:           {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)\n")
                    f.write(f"Precision:           {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)\n")
                    f.write(f"Recall:              {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)\n")
                    f.write(f"F1-Score:            {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)\n")
                    f.write(f"ROC-AUC:             {metrics['roc_auc']:.4f} ({metrics['roc_auc']*100:.2f}%)\n")
                    f.write(f"Specificity:         {metrics['specificity']:.4f} ({metrics['specificity']*100:.2f}%)\n")
                    f.write(f"False Positive Rate: {metrics['false_positive_rate']:.4f} ({metrics['false_positive_rate']*100:.2f}%)\n")
                    f.write(f"False Negative Rate: {metrics['false_negative_rate']:.4f} ({metrics['false_negative_rate']*100:.2f}%)\n")
                    
                    f.write(f"\nConfusion Matrix:\n")
                    f.write(f"TN: {metrics['tn']}, FP: {metrics['fp']}\n")
                    f.write(f"FN: {metrics['fn']}, TP: {metrics['tp']}\n")
                    
            # Comparison table
            f.write(f"\n\nMODEL COMPARISON TABLE\n")
            f.write("-" * 40 + "\n")
            f.write(comparison_df.to_string(index=False))
            
        print(f"Detailed report saved to: {report_path}")
        
    def run_metrics_generation(self, model_names=None):
        """
        Run complete metrics generation pipeline.
        
        Args:
            model_names: List of model names to analyze (None for all)
        """
        print("="*80)
        print("METRICS GENERATION PIPELINE")
        print("="*80)
        
        if model_names is None:
            model_names = self.models
            
        # Load test data
        self.load_test_data()
        
        # Generate reports for each model
        model_reports = []
        for model_name in model_names:
            report = self.generate_model_report(model_name)
            model_reports.append(report)
            
        # Generate comparison report
        comparison_df = self.generate_comparison_report(model_reports)
        
        # Generate plots
        self.plot_comparison_charts(model_reports)
        
        # Save detailed report
        self.save_detailed_report(model_reports, comparison_df)
        
        print(f"\n{'='*80}")
        print("METRICS GENERATION COMPLETED SUCCESSFULLY!")
        print(f"{'='*80}")
        
        return model_reports, comparison_df


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Generate comprehensive metrics report for SMS spam classification models')
    parser.add_argument('--results-dir', type=str, help='Path to results directory')
    parser.add_argument('--data-dir', type=str, help='Path to data directory')
    parser.add_argument('--model', type=str, choices=['naive_bayes', 'logistic_regression', 'random_forest'],
                       help='Generate report for specific model only')
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = MetricsGenerator(args.results_dir, args.data_dir)
    
    # Run metrics generation
    model_names = [args.model] if args.model else None
    generator.run_metrics_generation(model_names)


if __name__ == "__main__":
    main()
