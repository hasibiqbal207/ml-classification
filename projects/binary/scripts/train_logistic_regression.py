#!/usr/bin/env python3
"""
Logistic Regression Training Script for SMS Spam Binary Classification

This script trains a Logistic Regression model on the SMS spam dataset
and evaluates its performance.
"""

import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from algorithms.logistic_regression.sklearn_impl import LogisticRegressionSklearn


class SMSLogisticRegressionTrainer:
    """Trainer for SMS spam classification using Logistic Regression."""
    
    def __init__(self, data_dir: str = None, results_dir: str = None):
        """
        Initialize the trainer.
        
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
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.vectorizer = None
        self.model = None
        self.vocabulary = None
        
    def load_data(self):
        """Load processed data splits."""
        print("Loading processed data...")
        
        train_path = self.data_dir / "processed" / "train.csv"
        val_path = self.data_dir / "processed" / "val.csv"
        test_path = self.data_dir / "processed" / "test.csv"
        
        self.train_data = pd.read_csv(train_path)
        self.val_data = pd.read_csv(val_path)
        self.test_data = pd.read_csv(test_path)
        
        print(f"Training data: {len(self.train_data)} samples")
        print(f"Validation data: {len(self.val_data)} samples")
        print(f"Test data: {len(self.test_data)} samples")
        
        # Load vocabulary
        vocab_path = self.data_dir / "processed" / "vocabulary.pkl"
        with open(vocab_path, 'rb') as f:
            self.vocabulary = pickle.load(f)
            
        print(f"Vocabulary size: {len(self.vocabulary)}")
        
    def prepare_features(self, vectorizer_type='tfidf', max_features=None, ngram_range=(1, 2)):
        """
        Prepare text features using vectorization.
        
        Args:
            vectorizer_type: 'count' or 'tfidf'
            max_features: Maximum number of features
            ngram_range: Range of n-grams to use
        """
        print(f"Preparing features using {vectorizer_type} vectorizer...")
        
        if vectorizer_type == 'count':
            self.vectorizer = CountVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                vocabulary=self.vocabulary
            )
        elif vectorizer_type == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                vocabulary=self.vocabulary
            )
        else:
            raise ValueError("vectorizer_type must be 'count' or 'tfidf'")
            
        # Fit on training data
        self.X_train = self.vectorizer.fit_transform(self.train_data['text'])
        self.X_val = self.vectorizer.transform(self.val_data['text'])
        self.X_test = self.vectorizer.transform(self.test_data['text'])
        
        # Prepare labels
        self.y_train = self.train_data['label'].map({'ham': 0, 'spam': 1})
        self.y_val = self.val_data['label'].map({'ham': 0, 'spam': 1})
        self.y_test = self.test_data['label'].map({'ham': 0, 'spam': 1})
        
        print(f"Feature matrix shape - Train: {self.X_train.shape}, Val: {self.X_val.shape}, Test: {self.X_test.shape}")
        
    def train_model(self, C=1.0, penalty='l2', max_iter=1000):
        """
        Train the Logistic Regression model.
        
        Args:
            C: Regularization parameter
            penalty: 'l1' or 'l2' regularization
            max_iter: Maximum number of iterations
        """
        print(f"Training Logistic Regression model (C={C}, penalty={penalty})...")
        
        self.model = LogisticRegressionSklearn(
            random_state=42,
            max_iter=max_iter
        )
        
        # Update model parameters
        self.model.model.C = C
        self.model.model.penalty = penalty
        
        # Train the model
        self.model.fit(self.X_train.toarray(), self.y_train)
        
        print("Model training completed!")
        
    def hyperparameter_tuning(self):
        """Perform hyperparameter tuning using GridSearchCV."""
        print("\nPerforming hyperparameter tuning...")
        
        # Define parameter grid
        param_grid = {
            'C': [0.1, 1.0, 10.0, 100.0],
            'penalty': ['l1', 'l2'],
            'max_iter': [1000, 2000]
        }
        
        # Create base model
        base_model = LogisticRegressionSklearn(random_state=42)
        
        # Create GridSearchCV
        grid_search = GridSearchCV(
            base_model.model,
            param_grid,
            cv=3,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        # Fit on training data
        grid_search.fit(self.X_train.toarray(), self.y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        # Update model with best parameters
        self.model = LogisticRegressionSklearn(random_state=42)
        self.model.model.C = grid_search.best_params_['C']
        self.model.model.penalty = grid_search.best_params_['penalty']
        self.model.model.max_iter = grid_search.best_params_['max_iter']
        
        # Retrain with best parameters
        self.model.fit(self.X_train.toarray(), self.y_train)
        
        return grid_search.best_params_
        
    def evaluate_model(self, data_type='test'):
        """
        Evaluate the model performance.
        
        Args:
            data_type: 'val' or 'test'
        """
        if data_type == 'val':
            X, y = self.X_val, self.y_val
            data_name = "Validation"
        else:
            X, y = self.X_test, self.y_test
            data_name = "Test"
            
        print(f"\nEvaluating on {data_name} data...")
        
        # Make predictions
        y_pred = self.model.predict(X.toarray())
        y_proba = self.model.predict_proba(X.toarray())
        
        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        roc_auc = roc_auc_score(y, y_proba[:, 1])
        
        print(f"\n{data_name} Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        print(f"\nConfusion Matrix:")
        print(f"True Negatives: {cm[0,0]}, False Positives: {cm[0,1]}")
        print(f"False Negatives: {cm[1,0]}, True Positives: {cm[1,1]}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm
        }
        
    def cross_validate(self, cv=5):
        """Perform cross-validation."""
        print(f"\nPerforming {cv}-fold cross-validation...")
        
        # Combine train and val for CV
        X_combined = np.vstack([self.X_train.toarray(), self.X_val.toarray()])
        y_combined = np.hstack([self.y_train, self.y_val])
        
        # Use the underlying sklearn model for cross-validation
        from sklearn.linear_model import LogisticRegression
        cv_model = LogisticRegression(random_state=42, max_iter=1000)
        
        # Perform cross-validation
        cv_scores = cross_val_score(cv_model, X_combined, y_combined, cv=cv, scoring='accuracy')
        
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return cv_scores
        
    def plot_confusion_matrix(self, cm, title="Confusion Matrix"):
        """Plot confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        # Save plot
        plot_path = self.results_dir / f"logistic_regression_{title.lower().replace(' ', '_')}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_feature_importance(self, top_n=20):
        """Plot feature importance (coefficients)."""
        if not hasattr(self.model.model, 'coef_'):
            print("Model doesn't have coefficients (feature importance)")
            return
            
        # Get feature names
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Get coefficients
        coef = self.model.model.coef_[0]
        
        # Get top features
        top_indices = np.argsort(np.abs(coef))[-top_n:]
        top_features = feature_names[top_indices]
        top_coefs = coef[top_indices]
        
        # Plot
        plt.figure(figsize=(10, 8))
        colors = ['red' if c < 0 else 'blue' for c in top_coefs]
        plt.barh(range(len(top_features)), top_coefs, color=colors)
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel('Coefficient Value')
        plt.title(f'Top {top_n} Feature Importance (Logistic Regression)')
        plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        # Save plot
        plot_path = self.results_dir / "logistic_regression_feature_importance.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def save_model(self):
        """Save the trained model and vectorizer."""
        print("\nSaving model and vectorizer...")
        
        # Save model
        model_path = self.results_dir / "logistic_regression_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
            
        # Save vectorizer
        vectorizer_path = self.results_dir / "logistic_regression_vectorizer.pkl"
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
            
        print(f"Model saved to: {model_path}")
        print(f"Vectorizer saved to: {vectorizer_path}")
        
    def run_training_pipeline(self, vectorizer_type='tfidf', tune_hyperparams=True):
        """Run the complete training pipeline."""
        print("="*60)
        print("LOGISTIC REGRESSION TRAINING PIPELINE")
        print("="*60)
        
        # Load data
        self.load_data()
        
        # Prepare features
        self.prepare_features(vectorizer_type=vectorizer_type)
        
        # Hyperparameter tuning
        if tune_hyperparams:
            best_params = self.hyperparameter_tuning()
        else:
            self.train_model()
            
        # Evaluate on validation set
        val_results = self.evaluate_model('val')
        
        # Evaluate on test set
        test_results = self.evaluate_model('test')
        
        # Cross-validation
        cv_scores = self.cross_validate()
        
        # Plot results
        self.plot_confusion_matrix(val_results['confusion_matrix'], "Validation Confusion Matrix")
        self.plot_confusion_matrix(test_results['confusion_matrix'], "Test Confusion Matrix")
        self.plot_feature_importance()
        
        # Save model
        self.save_model()
        
        print("\n" + "="*60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        return {
            'validation_results': val_results,
            'test_results': test_results,
            'cv_scores': cv_scores
        }


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Train Logistic Regression model for SMS spam classification')
    parser.add_argument('--data-dir', type=str, help='Path to data directory')
    parser.add_argument('--results-dir', type=str, help='Path to results directory')
    parser.add_argument('--vectorizer', type=str, default='tfidf', choices=['count', 'tfidf'],
                       help='Type of vectorizer to use')
    parser.add_argument('--no-tuning', action='store_true', help='Skip hyperparameter tuning')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = SMSLogisticRegressionTrainer(args.data_dir, args.results_dir)
    
    # Run training pipeline
    results = trainer.run_training_pipeline(
        vectorizer_type=args.vectorizer,
        tune_hyperparams=not args.no_tuning
    )


if __name__ == "__main__":
    main()
