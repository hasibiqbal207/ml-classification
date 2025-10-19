#!/usr/bin/env python3
"""
Data preprocessing script for GoEmotions Multilabel Classification

This script processes the raw GoEmotions dataset and prepares it for multilabel classification.
It includes text cleaning, multilabel handling, and train/validation/test splits.
"""

import os
import sys
import csv
import re
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict
from collections import Counter
import logging

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GoEmotionsDataProcessor:
    """Data processor for GoEmotions multilabel classification dataset."""
    
    def __init__(self, data_dir: str = None):
        """
        Initialize the data processor.
        
        Args:
            data_dir: Path to the data directory
        """
        if data_dir is None:
            self.data_dir = Path(__file__).parent.parent / "data"
        else:
            self.data_dir = Path(data_dir)
        
        self.raw_data_path = self.data_dir / "raw" / "go_emotions_dataset.csv"
        self.processed_data_path = self.data_dir / "processed"
        
        # Create processed directory if it doesn't exist
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
        
        # Text preprocessing patterns
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.special_chars_pattern = re.compile(r'[^a-zA-Z0-9\s]')
        
        # Emotion categories (27 labels)
        self.emotion_categories = [
            'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
            'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
            'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
            'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
            'relief', 'remorse', 'sadness', 'surprise', 'neutral'
        ]
        
    def load_raw_data(self) -> pd.DataFrame:
        """
        Load raw GoEmotions data from CSV file.
        
        Returns:
            DataFrame with text and emotion labels
        """
        logger.info(f"Loading raw data from {self.raw_data_path}")
        
        try:
            df = pd.read_csv(self.raw_data_path)
            logger.info(f"Loaded {len(df)} comments")
            
            # Filter out unclear examples if desired
            # df = df[df['example_very_unclear'] == False]
            # logger.info(f"After filtering unclear examples: {len(df)} comments")
            
            return df
            
        except FileNotFoundError:
            logger.error(f"Raw data file not found: {self.raw_data_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading raw data: {e}")
            raise
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize Reddit comment text.
        
        Args:
            text: Raw Reddit comment text
            
        Returns:
            Cleaned text
        """
        if pd.isna(text):
            return ""
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Replace URLs with placeholder
        text = self.url_pattern.sub(' <url> ', text)
        
        # Replace email addresses with placeholder
        text = self.email_pattern.sub(' <email> ', text)
        
        # Remove Reddit-specific patterns
        text = re.sub(r'\[.*?\]', '', text)  # Remove [NAME], [RELIGION], etc.
        text = re.sub(r'u/[a-zA-Z0-9_]+', '<user>', text)  # Replace usernames
        text = re.sub(r'r/[a-zA-Z0-9_]+', '<subreddit>', text)  # Replace subreddits
        
        # Replace special characters and normalize whitespace
        text = self.special_chars_pattern.sub(' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the raw data.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        logger.info("Preprocessing data...")
        
        # Clean text
        df['text'] = df['text'].apply(self.clean_text)
        
        # Filter out empty texts
        df = df[df['text'].str.len() > 0]
        
        # Filter out very short texts (less than 3 words)
        df = df[df['text'].str.split().str.len() >= 3]
        
        logger.info(f"Preprocessed {len(df)} comments")
        return df
    
    def create_vocabulary(self, texts: List[str], min_freq: int = 5) -> Dict[str, int]:
        """
        Create vocabulary from texts.
        
        Args:
            texts: List of preprocessed texts
            min_freq: Minimum frequency for a word to be included
            
        Returns:
            Dictionary mapping words to indices
        """
        logger.info("Creating vocabulary...")
        
        # Count word frequencies
        word_counts = Counter()
        for text in texts:
            words = text.split()
            word_counts.update(words)
        
        # Create vocabulary with minimum frequency filter
        vocab = {'<PAD>': 0, '<UNK>': 1}  # Special tokens
        idx = 2
        
        for word, count in word_counts.most_common():
            if count >= min_freq:
                vocab[word] = idx
                idx += 1
        
        logger.info(f"Vocabulary size: {len(vocab)} (min_freq={min_freq})")
        return vocab
    
    def prepare_multilabel_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare multilabel data for training.
        
        Args:
            df: Preprocessed DataFrame
            
        Returns:
            Tuple of (texts, labels) where labels is a binary matrix
        """
        logger.info("Preparing multilabel data...")
        
        # Extract texts
        texts = df['text'].values
        
        # Extract emotion labels
        emotion_columns = [col for col in df.columns if col in self.emotion_categories]
        labels = df[emotion_columns].values.astype(int)
        
        # Log label statistics
        label_counts = labels.sum(axis=0)
        logger.info("Label distribution:")
        for i, emotion in enumerate(emotion_columns):
            logger.info(f"  {emotion}: {label_counts[i]} ({label_counts[i]/len(labels)*100:.2f}%)")
        
        # Log multilabel statistics
        labels_per_sample = labels.sum(axis=1)
        logger.info(f"Average labels per sample: {labels_per_sample.mean():.2f}")
        logger.info(f"Max labels per sample: {labels_per_sample.max()}")
        logger.info(f"Samples with 0 labels: {(labels_per_sample == 0).sum()}")
        logger.info(f"Samples with 1 label: {(labels_per_sample == 1).sum()}")
        logger.info(f"Samples with 2+ labels: {(labels_per_sample >= 2).sum()}")
        
        return texts, labels
    
    def split_data(self, texts: np.ndarray, labels: np.ndarray,
                   train_ratio: float = 0.7, 
                   val_ratio: float = 0.15, 
                   test_ratio: float = 0.15) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train/validation/test sets.
        
        Args:
            texts: Text data
            labels: Multilabel data
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            test_ratio: Ratio for test set
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        logger.info("Splitting data...")
        
        # Use stratified splitting based on label density
        from sklearn.model_selection import train_test_split
        
        # First split: train vs (val + test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            texts, labels, 
            test_size=(val_ratio + test_ratio), 
            random_state=42,
            stratify=None  # Multilabel stratification is complex, using random split
        )
        
        # Second split: val vs test
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=test_ratio/(val_ratio + test_ratio),
            random_state=42
        )
        
        logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def save_processed_data(self, X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray,
                           y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray,
                           vocab: Dict[str, int]) -> None:
        """
        Save processed data and vocabulary.
        
        Args:
            X_train, X_val, X_test: Text data splits
            y_train, y_val, y_test: Label data splits
            vocab: Vocabulary dictionary
        """
        logger.info("Saving processed data...")
        
        # Save text data
        text_splits = {
            'train': X_train,
            'val': X_val,
            'test': X_test
        }
        
        for split_name, text_data in text_splits.items():
            file_path = self.processed_data_path / f"{split_name}_texts.csv"
            pd.DataFrame({'text': text_data}).to_csv(file_path, index=False)
            logger.info(f"Saved {split_name} texts to {file_path}")
        
        # Save label data
        label_splits = {
            'train': y_train,
            'val': y_val,
            'test': y_test
        }
        
        for split_name, label_data in label_splits.items():
            file_path = self.processed_data_path / f"{split_name}_labels.csv"
            label_df = pd.DataFrame(label_data, columns=self.emotion_categories)
            label_df.to_csv(file_path, index=False)
            logger.info(f"Saved {split_name} labels to {file_path}")
        
        # Save vocabulary
        vocab_path = self.processed_data_path / "vocabulary.pkl"
        with open(vocab_path, 'wb') as f:
            pickle.dump(vocab, f)
        logger.info(f"Saved vocabulary to {vocab_path}")
        
        # Save vocabulary as text file for inspection
        vocab_text_path = self.processed_data_path / "vocabulary.txt"
        with open(vocab_text_path, 'w', encoding='utf-8') as f:
            for word, idx in sorted(vocab.items(), key=lambda x: x[1]):
                f.write(f"{word}\t{idx}\n")
        logger.info(f"Saved vocabulary text to {vocab_text_path}")
        
        # Save emotion categories
        categories_path = self.processed_data_path / "emotion_categories.pkl"
        with open(categories_path, 'wb') as f:
            pickle.dump(self.emotion_categories, f)
        logger.info(f"Saved emotion categories to {categories_path}")
    
    def generate_statistics(self, y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Generate dataset statistics.
        
        Args:
            y_train, y_val, y_test: Label data splits
            
        Returns:
            Dictionary with statistics
        """
        stats = {}
        
        for split_name, split_labels in [('train', y_train), ('val', y_val), ('test', y_test)]:
            # Calculate label statistics
            label_counts = split_labels.sum(axis=0)
            labels_per_sample = split_labels.sum(axis=1)
            
            stats[split_name] = {
                'total_samples': len(split_labels),
                'total_labels': split_labels.sum(),
                'avg_labels_per_sample': labels_per_sample.mean(),
                'max_labels_per_sample': labels_per_sample.max(),
                'samples_with_0_labels': (labels_per_sample == 0).sum(),
                'samples_with_1_label': (labels_per_sample == 1).sum(),
                'samples_with_2_plus_labels': (labels_per_sample >= 2).sum(),
                'label_distribution': dict(zip(self.emotion_categories, label_counts.tolist()))
            }
        
        return stats
    
    def process(self, min_freq: int = 5, train_ratio: float = 0.7, 
               val_ratio: float = 0.15, test_ratio: float = 0.15) -> Dict:
        """
        Main processing pipeline.
        
        Args:
            min_freq: Minimum word frequency for vocabulary
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            
        Returns:
            Dictionary with processing statistics
        """
        logger.info("Starting data processing pipeline...")
        
        # Load raw data
        raw_df = self.load_raw_data()
        
        # Preprocess data
        processed_df = self.preprocess_data(raw_df)
        
        # Prepare multilabel data
        texts, labels = self.prepare_multilabel_data(processed_df)
        
        # Create vocabulary
        vocab = self.create_vocabulary(texts, min_freq)
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(
            texts, labels, train_ratio, val_ratio, test_ratio
        )
        
        # Save processed data
        self.save_processed_data(X_train, X_val, X_test, y_train, y_val, y_test, vocab)
        
        # Generate statistics
        stats = self.generate_statistics(y_train, y_val, y_test)
        
        logger.info("Data processing completed successfully!")
        return stats


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Preprocess GoEmotions multilabel dataset')
    parser.add_argument('--data-dir', type=str, help='Path to data directory')
    parser.add_argument('--min-freq', type=int, default=5, help='Minimum word frequency')
    parser.add_argument('--train-ratio', type=float, default=0.7, help='Training set ratio')
    parser.add_argument('--val-ratio', type=float, default=0.15, help='Validation set ratio')
    parser.add_argument('--test-ratio', type=float, default=0.15, help='Test set ratio')
    
    args = parser.parse_args()
    
    # Validate ratios
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        logger.error("Ratios must sum to 1.0")
        sys.exit(1)
    
    # Initialize processor
    processor = GoEmotionsDataProcessor(args.data_dir)
    
    # Process data
    stats = processor.process(
        min_freq=args.min_freq,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )
    
    # Print statistics
    print("\n" + "="*50)
    print("PROCESSING STATISTICS")
    print("="*50)
    
    for split_name, split_stats in stats.items():
        print(f"\n{split_name.upper()} SET:")
        print(f"  Total samples: {split_stats['total_samples']}")
        print(f"  Total labels: {split_stats['total_labels']}")
        print(f"  Average labels per sample: {split_stats['avg_labels_per_sample']:.2f}")
        print(f"  Max labels per sample: {split_stats['max_labels_per_sample']}")
        print(f"  Samples with 0 labels: {split_stats['samples_with_0_labels']}")
        print(f"  Samples with 1 label: {split_stats['samples_with_1_label']}")
        print(f"  Samples with 2+ labels: {split_stats['samples_with_2_plus_labels']}")
        
        print(f"  Top 5 emotion categories:")
        sorted_emotions = sorted(split_stats['label_distribution'].items(), 
                               key=lambda x: x[1], reverse=True)[:5]
        for emotion, count in sorted_emotions:
            print(f"    {emotion}: {count}")


if __name__ == "__main__":
    main()
