#!/usr/bin/env python3
"""
RNN-specific data preprocessing for GoEmotions multilabel classification.

This script prepares data specifically for RNN models (LSTM, BiLSTM, GRU)
with proper text tokenization and sequence creation.
"""

import os
import sys
import pickle
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def preprocess_text(text: str) -> str:
    """
    Preprocess text for RNN models.
    
    Args:
        text: Raw text string
        
    Returns:
        Preprocessed text string
    """
    if pd.isna(text) or text == '':
        return ''
    
    # Convert to lowercase
    text = str(text).lower()
    
    # Replace URLs with placeholder
    url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    text = url_pattern.sub(' <url> ', text)
    
    # Replace email addresses with placeholder
    email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    text = email_pattern.sub(' <email> ', text)
    
    # Remove Reddit-specific patterns
    text = re.sub(r'\[.*?\]', '', text)  # Remove [NAME], [RELIGION], etc.
    text = re.sub(r'u/[a-zA-Z0-9_]+', '<user>', text)  # Replace usernames
    text = re.sub(r'r/[a-zA-Z0-9_]+', '<subreddit>', text)  # Replace subreddits
    
    # Replace special characters and normalize whitespace
    special_chars_pattern = re.compile(r'[^a-zA-Z0-9\s]')
    text = special_chars_pattern.sub(' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def load_and_preprocess_data(data_path: str) -> Tuple[List[str], List[List[str]]]:
    """
    Load and preprocess GoEmotions dataset for RNN training.
    
    Args:
        data_path: Path to the raw dataset
        
    Returns:
        Tuple of (texts, emotion_labels)
    """
    logger.info("Loading GoEmotions dataset...")
    
    # Load dataset
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} samples")
    
    # Define emotion columns (excluding id, text, example_very_unclear)
    emotion_columns = [col for col in df.columns if col not in ['id', 'text', 'example_very_unclear']]
    logger.info(f"Found {len(emotion_columns)} emotion categories: {emotion_columns}")
    
    # Preprocess texts
    logger.info("Preprocessing texts...")
    texts = []
    emotion_labels = []
    
    for idx, row in df.iterrows():
        if idx % 10000 == 0:
            logger.info(f"Processed {idx}/{len(df)} samples")
        
        # Preprocess text
        text = preprocess_text(row['text'])
        if text == '' or len(text.split()) < 2:  # Skip very short texts
            continue
        
        # Get emotions for this sample (columns with value 1)
        emotions = [emotion for emotion in emotion_columns if row[emotion] == 1]
        
        texts.append(text)
        emotion_labels.append(emotions)
    
    logger.info(f"Preprocessed {len(texts)} valid samples")
    return texts, emotion_labels

def create_tokenizer(texts: List[str], vocab_size: int = 10000) -> Tokenizer:
    """
    Create and fit tokenizer for text sequences.
    
    Args:
        texts: List of preprocessed texts
        vocab_size: Maximum vocabulary size
        
    Returns:
        Fitted tokenizer
    """
    logger.info(f"Creating tokenizer with vocab_size={vocab_size}")
    
    tokenizer = Tokenizer(
        num_words=vocab_size,
        oov_token='<OOV>',
        filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    )
    
    tokenizer.fit_on_texts(texts)
    
    logger.info(f"Tokenizer created with {len(tokenizer.word_index)} unique words")
    return tokenizer

def create_sequences(
    texts: List[str],
    tokenizer: Tokenizer,
    max_length: int = 100
) -> np.ndarray:
    """
    Convert texts to padded sequences for RNN input.
    
    Args:
        texts: List of preprocessed texts
        tokenizer: Fitted tokenizer
        max_length: Maximum sequence length
        
    Returns:
        Padded sequences array
    """
    logger.info(f"Creating sequences with max_length={max_length}")
    
    # Convert texts to sequences
    sequences = tokenizer.texts_to_sequences(texts)
    
    # Pad sequences
    padded_sequences = pad_sequences(
        sequences,
        maxlen=max_length,
        padding='post',
        truncating='post'
    )
    
    logger.info(f"Created sequences with shape: {padded_sequences.shape}")
    return padded_sequences

def prepare_multilabel_data(
    emotion_labels: List[List[str]],
    emotion_categories: List[str]
) -> np.ndarray:
    """
    Prepare multilabel data for RNN training.
    
    Args:
        emotion_labels: List of emotion label lists
        emotion_categories: List of all emotion categories
        
    Returns:
        Binary matrix of emotion labels
    """
    logger.info("Preparing multilabel data...")
    
    # Create multilabel binarizer
    mlb = MultiLabelBinarizer()
    mlb.fit([emotion_categories])
    
    # Transform labels to binary matrix
    y_binary = mlb.transform(emotion_labels)
    
    logger.info(f"Created multilabel matrix with shape: {y_binary.shape}")
    return y_binary, mlb

def split_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train, validation, and test sets.
    
    Args:
        X: Input sequences
        y: Target labels
        test_size: Test set size
        val_size: Validation set size
        random_state: Random seed
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    logger.info("Splitting data into train/validation/test sets...")
    
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=None
    )
    
    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=None
    )
    
    logger.info(f"Data split completed:")
    logger.info(f"  Train: {X_train.shape[0]} samples")
    logger.info(f"  Validation: {X_val.shape[0]} samples")
    logger.info(f"  Test: {X_test.shape[0]} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def save_processed_data(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    tokenizer: Tokenizer,
    mlb: MultiLabelBinarizer,
    emotion_categories: List[str],
    output_dir: str
):
    """
    Save processed data and tokenizer.
    
    Args:
        X_train, X_val, X_test: Input sequences
        y_train, y_val, y_test: Target labels
        tokenizer: Fitted tokenizer
        mlb: Multilabel binarizer
        emotion_categories: List of emotion categories
        output_dir: Output directory
    """
    logger.info(f"Saving processed data to {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save sequences
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(output_dir, 'X_val.npy'), X_val)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
    
    # Save labels
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'y_val.npy'), y_val)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
    
    # Save tokenizer
    with open(os.path.join(output_dir, 'tokenizer.pkl'), 'wb') as f:
        pickle.dump(tokenizer, f)
    
    # Save multilabel binarizer
    with open(os.path.join(output_dir, 'mlb.pkl'), 'wb') as f:
        pickle.dump(mlb, f)
    
    # Save emotion categories
    with open(os.path.join(output_dir, 'emotion_categories.pkl'), 'wb') as f:
        pickle.dump(emotion_categories, f)
    
    # Save vocabulary
    vocab = list(tokenizer.word_index.keys())[:tokenizer.num_words]
    with open(os.path.join(output_dir, 'vocabulary.txt'), 'w') as f:
        for word in vocab:
            f.write(f"{word}\n")
    
    logger.info("Data saved successfully")

def main():
    """Main preprocessing function."""
    # Configuration
    data_path = "data/raw/go_emotions_dataset.csv"
    output_dir = "data/processed_rnn"
    vocab_size = 10000
    max_length = 100
    
    # Emotion categories will be determined from the dataset columns
    emotion_categories = None
    
    logger.info("Starting RNN data preprocessing...")
    
    try:
        # Load and preprocess data
        texts, emotion_labels = load_and_preprocess_data(data_path)
        
        # Get emotion categories from the data
        emotion_categories = list(set([emotion for emotions in emotion_labels for emotion in emotions]))
        emotion_categories.sort()
        logger.info(f"Found {len(emotion_categories)} emotion categories: {emotion_categories}")
        
        # Create tokenizer
        tokenizer = create_tokenizer(texts, vocab_size)
        
        # Create sequences
        X = create_sequences(texts, tokenizer, max_length)
        
        # Prepare multilabel data
        y, mlb = prepare_multilabel_data(emotion_labels, emotion_categories)
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
        
        # Save processed data
        save_processed_data(
            X_train, X_val, X_test, y_train, y_val, y_test,
            tokenizer, mlb, emotion_categories, output_dir
        )
        
        logger.info("RNN data preprocessing completed successfully!")
        
        # Print summary
        print("\n" + "="*60)
        print("RNN DATA PREPROCESSING SUMMARY")
        print("="*60)
        print(f"Total samples: {len(texts)}")
        print(f"Vocabulary size: {len(tokenizer.word_index)}")
        print(f"Sequence length: {max_length}")
        print(f"Number of emotions: {len(emotion_categories)}")
        print(f"Train samples: {X_train.shape[0]}")
        print(f"Validation samples: {X_val.shape[0]}")
        print(f"Test samples: {X_test.shape[0]}")
        print(f"Output directory: {output_dir}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise

if __name__ == "__main__":
    main()
