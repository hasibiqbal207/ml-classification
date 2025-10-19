#!/usr/bin/env python3
"""
RNN model training script for GoEmotions multilabel classification.

This script trains LSTM, BiLSTM, and GRU models for emotion analysis.
"""

import os
import sys
import pickle
import logging
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from algorithms.rnn.tensorflow_impl import (
    LSTMEmotionClassifier, BiLSTMEmotionClassifier, GRUEmotionClassifier
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_processed_data(data_dir: str) -> Dict[str, Any]:
    """
    Load preprocessed RNN data.
    
    Args:
        data_dir: Directory containing processed data
        
    Returns:
        Dictionary containing loaded data
    """
    logger.info(f"Loading processed data from {data_dir}")
    
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
    
    with open(os.path.join(data_dir, 'mlb.pkl'), 'rb') as f:
        mlb = pickle.load(f)
    
    with open(os.path.join(data_dir, 'emotion_categories.pkl'), 'rb') as f:
        emotion_categories = pickle.load(f)
    
    logger.info(f"Loaded data shapes:")
    logger.info(f"  X_train: {X_train.shape}")
    logger.info(f"  X_val: {X_val.shape}")
    logger.info(f"  X_test: {X_test.shape}")
    logger.info(f"  y_train: {y_train.shape}")
    logger.info(f"  y_val: {y_val.shape}")
    logger.info(f"  y_test: {y_test.shape}")
    
    return {
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
        'tokenizer': tokenizer, 'mlb': mlb, 'emotion_categories': emotion_categories
    }

def train_rnn_model(
    model_type: str,
    data: Dict[str, Any],
    model_params: Dict[str, Any],
    training_params: Dict[str, Any],
    output_dir: str
) -> Dict[str, Any]:
    """
    Train a single RNN model.
    
    Args:
        model_type: Type of RNN model ('lstm', 'bilstm', 'gru')
        data: Dictionary containing training data
        model_params: Model hyperparameters
        training_params: Training hyperparameters
        output_dir: Output directory for saving model
        
    Returns:
        Dictionary containing training results
    """
    logger.info(f"Training {model_type.upper()} model...")
    
    # Create model
    if model_type.lower() == 'lstm':
        model = LSTMEmotionClassifier(**model_params)
    elif model_type.lower() == 'bilstm':
        model = BiLSTMEmotionClassifier(**model_params)
    elif model_type.lower() == 'gru':
        model = GRUEmotionClassifier(**model_params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Prepare callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=training_params.get('early_stopping_patience', 5),
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=training_params.get('lr_reduction_factor', 0.5),
            patience=training_params.get('lr_reduction_patience', 3),
            min_lr=training_params.get('min_lr', 1e-7),
            verbose=1
        )
    ]
    
    # Add model checkpoint if specified
    if training_params.get('save_best_model', True):
        checkpoint_path = os.path.join(output_dir, f'{model_type}_best_model.h5')
        callbacks.append(
            ModelCheckpoint(
                checkpoint_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        )
    
    # Train model
    history = model.fit(
        data['X_train'], data['y_train'],
        validation_data=(data['X_val'], data['y_val']),
        epochs=training_params.get('epochs', 20),
        batch_size=training_params.get('batch_size', 32),
        verbose=1,
        callbacks=callbacks
    )
    
    # Evaluate model
    logger.info("Evaluating model...")
    train_metrics = model.evaluate(data['X_train'], data['y_train'])
    val_metrics = model.evaluate(data['X_val'], data['y_val'])
    test_metrics = model.evaluate(data['X_test'], data['y_test'])
    
    # Save model
    model_path = os.path.join(output_dir, f'{model_type}_model.h5')
    model.save_model(model_path)
    
    # Save tokenizer and metadata
    tokenizer_path = os.path.join(output_dir, f'{model_type}_tokenizer.pkl')
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(data['tokenizer'], f)
    
    mlb_path = os.path.join(output_dir, f'{model_type}_mlb.pkl')
    with open(mlb_path, 'wb') as f:
        pickle.dump(data['mlb'], f)
    
    emotion_categories_path = os.path.join(output_dir, f'{model_type}_emotion_categories.pkl')
    with open(emotion_categories_path, 'wb') as f:
        pickle.dump(data['emotion_categories'], f)
    
    # Plot training history
    history_plot_path = os.path.join(output_dir, f'{model_type}_training_history.png')
    model.plot_training_history(history_plot_path)
    
    logger.info(f"{model_type.upper()} training completed!")
    logger.info(f"Best validation loss: {min(history.history['val_loss']):.4f}")
    logger.info(f"Test F1-macro: {test_metrics['f1_macro']:.4f}")
    logger.info(f"Test Hamming Loss: {test_metrics['hamming_loss']:.4f}")
    
    return {
        'model_type': model_type,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'history': history.history,
        'model_path': model_path
    }

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train RNN models for emotion classification')
    parser.add_argument('--data-dir', type=str, default='data/processed_rnn',
                       help='Directory containing processed data')
    parser.add_argument('--output-dir', type=str, default='results/rnn_models',
                       help='Output directory for trained models')
    parser.add_argument('--models', nargs='+', default=['lstm', 'bilstm', 'gru'],
                       choices=['lstm', 'bilstm', 'gru'],
                       help='RNN models to train')
    parser.add_argument('--vocab-size', type=int, default=10000,
                       help='Vocabulary size')
    parser.add_argument('--embedding-dim', type=int, default=128,
                       help='Embedding dimension')
    parser.add_argument('--rnn-units', type=int, default=64,
                       help='Number of RNN units')
    parser.add_argument('--max-length', type=int, default=100,
                       help='Maximum sequence length')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--dropout-rate', type=float, default=0.3,
                       help='Dropout rate')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    data = load_processed_data(args.data_dir)
    
    # Model parameters
    model_params = {
        'vocab_size': args.vocab_size,
        'embedding_dim': args.embedding_dim,
        'rnn_units': args.rnn_units,
        'num_emotions': len(data['emotion_categories']),
        'max_sequence_length': args.max_length,
        'dropout_rate': args.dropout_rate,
        'learning_rate': args.learning_rate
    }
    
    # Training parameters
    training_params = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'early_stopping_patience': 5,
        'lr_reduction_factor': 0.5,
        'lr_reduction_patience': 3,
        'min_lr': 1e-7,
        'save_best_model': True
    }
    
    # Train models
    results = {}
    for model_type in args.models:
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"TRAINING {model_type.upper()} MODEL")
            logger.info(f"{'='*60}")
            
            result = train_rnn_model(
                model_type, data, model_params, training_params, args.output_dir
            )
            results[model_type] = result
            
        except Exception as e:
            logger.error(f"Failed to train {model_type}: {e}")
            results[model_type] = {'error': str(e)}
    
    # Print summary
    print("\n" + "="*80)
    print("RNN TRAINING SUMMARY")
    print("="*80)
    
    for model_type, result in results.items():
        if 'error' in result:
            print(f"{model_type.upper()}: FAILED - {result['error']}")
        else:
            test_metrics = result['test_metrics']
            print(f"{model_type.upper()}:")
            print(f"  F1-macro: {test_metrics['f1_macro']:.4f}")
            print(f"  F1-micro: {test_metrics['f1_micro']:.4f}")
            print(f"  Hamming Loss: {test_metrics['hamming_loss']:.4f}")
            print(f"  Jaccard Score: {test_metrics['jaccard_score']:.4f}")
            print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    
    print("="*80)
    print(f"Models saved to: {args.output_dir}")
    print("="*80)

if __name__ == "__main__":
    main()
