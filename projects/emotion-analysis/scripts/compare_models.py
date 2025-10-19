#!/usr/bin/env python3
"""
Model comparison script for emotion classification.

This script compares traditional ML models with RNN models on the GoEmotions dataset.
"""

import os
import sys
import pickle
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Tuple
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from algorithms.rnn.tensorflow_impl import LSTMEmotionClassifier, BiLSTMEmotionClassifier, GRUEmotionClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_traditional_ml_data(data_dir: str) -> Dict[str, Any]:
    """Load data for traditional ML models."""
    logger.info("Loading traditional ML data...")
    
    # Load text data
    texts_train = pd.read_csv(os.path.join(data_dir, 'train_texts.csv'))['text'].tolist()
    texts_val = pd.read_csv(os.path.join(data_dir, 'val_texts.csv'))['text'].tolist()
    texts_test = pd.read_csv(os.path.join(data_dir, 'test_texts.csv'))['text'].tolist()
    
    # Load labels
    y_train = pd.read_csv(os.path.join(data_dir, 'train_labels.csv')).values
    y_val = pd.read_csv(os.path.join(data_dir, 'val_labels.csv')).values
    y_test = pd.read_csv(os.path.join(data_dir, 'test_labels.csv')).values
    
    # Load emotion categories
    with open(os.path.join(data_dir, 'emotion_categories.pkl'), 'rb') as f:
        emotion_categories = pickle.load(f)
    
    return {
        'texts_train': texts_train, 'texts_val': texts_val, 'texts_test': texts_test,
        'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
        'emotion_categories': emotion_categories
    }

def load_rnn_data(data_dir: str) -> Dict[str, Any]:
    """Load data for RNN models."""
    logger.info("Loading RNN data...")
    
    # Load sequences
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    X_val = np.load(os.path.join(data_dir, 'X_val.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    
    # Load labels
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    y_val = np.load(os.path.join(data_dir, 'y_val.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    
    # Load tokenizer and metadata
    with open(os.path.join(data_dir, 'tokenizer.pkl'), 'rb') as f:
        tokenizer = pickle.load(f)
    
    with open(os.path.join(data_dir, 'emotion_categories.pkl'), 'rb') as f:
        emotion_categories = pickle.load(f)
    
    return {
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
        'tokenizer': tokenizer, 'emotion_categories': emotion_categories
    }

def train_traditional_ml_models(data: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
    """Train traditional ML models."""
    logger.info("Training traditional ML models...")
    
    results = {}
    
    # Vectorize texts
    vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
    X_train_vec = vectorizer.fit_transform(data['texts_train'])
    X_test_vec = vectorizer.transform(data['texts_test'])
    
    # Define models
    models = {
        'naive_bayes': OneVsRestClassifier(MultinomialNB()),
        'logistic_regression': OneVsRestClassifier(LogisticRegression(random_state=42, max_iter=1000)),
        'random_forest': OneVsRestClassifier(RandomForestClassifier(n_estimators=100, random_state=42)),
        'svm': OneVsRestClassifier(SVC(probability=True, random_state=42))
    }
    
    for name, model in models.items():
        try:
            logger.info(f"Training {name}...")
            
            # Train model
            model.fit(X_train_vec, data['y_train'])
            
            # Make predictions
            y_pred = model.predict(X_test_vec)
            y_proba = model.predict_proba(X_test_vec)
            
            # Calculate metrics
            from sklearn.metrics import (
                accuracy_score, f1_score, hamming_loss, jaccard_score
            )
            
            metrics = {
                'accuracy': accuracy_score(data['y_test'], y_pred),
                'f1_macro': f1_score(data['y_test'], y_pred, average='macro', zero_division=0),
                'f1_micro': f1_score(data['y_test'], y_pred, average='micro', zero_division=0),
                'hamming_loss': hamming_loss(data['y_test'], y_pred),
                'jaccard_score': jaccard_score(data['y_test'], y_pred, average='macro', zero_division=0)
            }
            
            results[name] = {
                'model': model,
                'vectorizer': vectorizer,
                'metrics': metrics,
                'predictions': y_pred,
                'probabilities': y_proba
            }
            
            logger.info(f"{name} - F1-macro: {metrics['f1_macro']:.4f}, Hamming Loss: {metrics['hamming_loss']:.4f}")
            
        except Exception as e:
            logger.error(f"Failed to train {name}: {e}")
            results[name] = {'error': str(e)}
    
    return results

def load_rnn_models(models_dir: str) -> Dict[str, Any]:
    """Load trained RNN models."""
    logger.info("Loading RNN models...")
    
    results = {}
    model_types = ['lstm', 'bilstm', 'gru']
    
    for model_type in model_types:
        try:
            model_path = os.path.join(models_dir, f'{model_type}_model.h5')
            if os.path.exists(model_path):
                # Load model
                if model_type == 'lstm':
                    model = LSTMEmotionClassifier()
                elif model_type == 'bilstm':
                    model = BiLSTMEmotionClassifier()
                elif model_type == 'gru':
                    model = GRUEmotionClassifier()
                
                model.load_model(model_path)
                
                # Load tokenizer
                tokenizer_path = os.path.join(models_dir, f'{model_type}_tokenizer.pkl')
                with open(tokenizer_path, 'rb') as f:
                    tokenizer = pickle.load(f)
                
                results[model_type] = {
                    'model': model,
                    'tokenizer': tokenizer
                }
                
                logger.info(f"Loaded {model_type} model")
            else:
                logger.warning(f"Model file not found: {model_path}")
                
        except Exception as e:
            logger.error(f"Failed to load {model_type}: {e}")
            results[model_type] = {'error': str(e)}
    
    return results

def evaluate_rnn_models(rnn_models: Dict[str, Any], rnn_data: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate RNN models."""
    logger.info("Evaluating RNN models...")
    
    results = {}
    
    for model_type, model_data in rnn_models.items():
        if 'error' in model_data:
            results[model_type] = model_data
            continue
        
        try:
            model = model_data['model']
            
            # Evaluate on test set
            test_metrics = model.evaluate(
                rnn_data['X_test'], 
                rnn_data['y_test'],
                emotion_categories=rnn_data['emotion_categories']
            )
            
            # Get predictions
            y_pred = model.predict(rnn_data['X_test'])
            y_proba = model.predict_proba(rnn_data['X_test'])
            
            results[model_type] = {
                'model': model,
                'metrics': test_metrics,
                'predictions': y_pred,
                'probabilities': y_proba
            }
            
            logger.info(f"{model_type} - F1-macro: {test_metrics['f1_macro']:.4f}, Hamming Loss: {test_metrics['hamming_loss']:.4f}")
            
        except Exception as e:
            logger.error(f"Failed to evaluate {model_type}: {e}")
            results[model_type] = {'error': str(e)}
    
    return results

def create_comparison_plots(
    traditional_results: Dict[str, Any],
    rnn_results: Dict[str, Any],
    output_dir: str
):
    """Create comparison plots."""
    logger.info("Creating comparison plots...")
    
    # Prepare data for plotting
    plot_data = []
    
    # Traditional ML results
    for name, result in traditional_results.items():
        if 'error' not in result:
            plot_data.append({
                'Model': name.replace('_', ' ').title(),
                'Type': 'Traditional ML',
                'F1-macro': result['metrics']['f1_macro'],
                'F1-micro': result['metrics']['f1_micro'],
                'Hamming Loss': result['metrics']['hamming_loss'],
                'Jaccard Score': result['metrics']['jaccard_score']
            })
    
    # RNN results
    for name, result in rnn_results.items():
        if 'error' not in result:
            plot_data.append({
                'Model': name.upper(),
                'Type': 'RNN',
                'F1-macro': result['metrics']['f1_macro'],
                'F1-micro': result['metrics']['f1_micro'],
                'Hamming Loss': result['metrics']['hamming_loss'],
                'Jaccard Score': result['metrics']['jaccard_score']
            })
    
    df = pd.DataFrame(plot_data)
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # F1-macro comparison
    sns.barplot(data=df, x='Model', y='F1-macro', hue='Type', ax=axes[0, 0])
    axes[0, 0].set_title('F1-macro Score Comparison')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # F1-micro comparison
    sns.barplot(data=df, x='Model', y='F1-micro', hue='Type', ax=axes[0, 1])
    axes[0, 1].set_title('F1-micro Score Comparison')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Hamming Loss comparison
    sns.barplot(data=df, x='Model', y='Hamming Loss', hue='Type', ax=axes[1, 0])
    axes[1, 0].set_title('Hamming Loss Comparison (Lower is Better)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Jaccard Score comparison
    sns.barplot(data=df, x='Model', y='Jaccard Score', hue='Type', ax=axes[1, 1])
    axes[1, 1].set_title('Jaccard Score Comparison')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create summary table
    summary_df = df.pivot_table(
        index='Model', 
        columns='Type', 
        values=['F1-macro', 'F1-micro', 'Hamming Loss', 'Jaccard Score'],
        aggfunc='first'
    )
    
    summary_df.to_csv(os.path.join(output_dir, 'model_comparison_summary.csv'))
    
    logger.info("Comparison plots created successfully")

def print_comparison_summary(traditional_results: Dict[str, Any], rnn_results: Dict[str, Any]):
    """Print comparison summary."""
    print("\n" + "="*80)
    print("MODEL COMPARISON SUMMARY")
    print("="*80)
    
    print("\nTRADITIONAL ML MODELS:")
    print("-" * 40)
    for name, result in traditional_results.items():
        if 'error' not in result:
            metrics = result['metrics']
            print(f"{name.replace('_', ' ').title():20} | F1-macro: {metrics['f1_macro']:.4f} | Hamming Loss: {metrics['hamming_loss']:.4f}")
        else:
            print(f"{name.replace('_', ' ').title():20} | ERROR: {result['error']}")
    
    print("\nRNN MODELS:")
    print("-" * 40)
    for name, result in rnn_results.items():
        if 'error' not in result:
            metrics = result['metrics']
            print(f"{name.upper():20} | F1-macro: {metrics['f1_macro']:.4f} | Hamming Loss: {metrics['hamming_loss']:.4f}")
        else:
            print(f"{name.upper():20} | ERROR: {result['error']}")
    
    print("="*80)

def main():
    """Main comparison function."""
    parser = argparse.ArgumentParser(description='Compare traditional ML and RNN models')
    parser.add_argument('--traditional-data-dir', type=str, default='data/processed',
                       help='Directory containing traditional ML processed data')
    parser.add_argument('--rnn-data-dir', type=str, default='data/processed_rnn',
                       help='Directory containing RNN processed data')
    parser.add_argument('--rnn-models-dir', type=str, default='results/rnn_models',
                       help='Directory containing trained RNN models')
    parser.add_argument('--output-dir', type=str, default='results/comparison',
                       help='Output directory for comparison results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Load data
        traditional_data = load_traditional_ml_data(args.traditional_data_dir)
        rnn_data = load_rnn_data(args.rnn_data_dir)
        
        # Train traditional ML models
        traditional_results = train_traditional_ml_models(traditional_data, args.output_dir)
        
        # Load RNN models
        rnn_models = load_rnn_models(args.rnn_models_dir)
        rnn_results = evaluate_rnn_models(rnn_models, rnn_data)
        
        # Create comparison plots
        create_comparison_plots(traditional_results, rnn_results, args.output_dir)
        
        # Print summary
        print_comparison_summary(traditional_results, rnn_results)
        
        logger.info(f"Comparison completed! Results saved to {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        raise

if __name__ == "__main__":
    main()
