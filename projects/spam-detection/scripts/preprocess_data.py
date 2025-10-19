#!/usr/bin/env python3
"""
Data preprocessing script for SMS Spam Binary Classification

This script processes the raw SMS spam dataset and prepares it for training.
It includes text cleaning, tokenization, and train/validation/test splits.
"""

import os
import sys
import csv
import re
import pickle
import argparse
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


class SMSDataProcessor:
    """Data processor for SMS spam classification dataset."""
    
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
        
        self.raw_data_path = self.data_dir / "raw" / "spam.csv"
        self.processed_data_path = self.data_dir / "processed"
        
        # Create processed directory if it doesn't exist
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
        
        # Text preprocessing patterns
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.phone_pattern = re.compile(r'\b\d{10,}\b')
        self.special_chars_pattern = re.compile(r'[^a-zA-Z0-9\s]')
        
    def load_raw_data(self) -> List[Tuple[str, str]]:
        """
        Load raw SMS data from CSV file.
        
        Returns:
            List of (label, text) tuples
        """
        logger.info(f"Loading raw data from {self.raw_data_path}")
        
        data = []
        try:
            with open(self.raw_data_path, 'r', encoding='latin-1') as f:
                reader = csv.reader(f)
                header = next(reader)  # Skip header
                
                for row in reader:
                    if len(row) >= 2 and row[0] and row[1]:
                        label = row[0].strip()
                        text = row[1].strip()
                        if label in ['ham', 'spam'] and text:
                            data.append((label, text))
                            
        except FileNotFoundError:
            logger.error(f"Raw data file not found: {self.raw_data_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading raw data: {e}")
            raise
            
        logger.info(f"Loaded {len(data)} messages")
        return data
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize SMS text.
        
        Args:
            text: Raw SMS text
            
        Returns:
            Cleaned text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Replace URLs with placeholder
        text = self.url_pattern.sub(' <url> ', text)
        
        # Replace email addresses with placeholder
        text = self.email_pattern.sub(' <email> ', text)
        
        # Replace phone numbers with placeholder
        text = self.phone_pattern.sub(' <phone> ', text)
        
        # Replace special characters and normalize whitespace
        text = self.special_chars_pattern.sub(' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def preprocess_data(self, data: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """
        Preprocess the raw data.
        
        Args:
            data: List of (label, text) tuples
            
        Returns:
            List of preprocessed (label, text) tuples
        """
        logger.info("Preprocessing data...")
        
        processed_data = []
        for label, text in data:
            cleaned_text = self.clean_text(text)
            if cleaned_text:  # Only keep non-empty texts
                processed_data.append((label, cleaned_text))
        
        logger.info(f"Preprocessed {len(processed_data)} messages")
        return processed_data
    
    def create_vocabulary(self, texts: List[str], min_freq: int = 2) -> Dict[str, int]:
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
    
    def split_data(self, data: List[Tuple[str, str]], 
                   train_ratio: float = 0.7, 
                   val_ratio: float = 0.15, 
                   test_ratio: float = 0.15,
                   stratify: bool = True) -> Tuple[List, List, List]:
        """
        Split data into train/validation/test sets.
        
        Args:
            data: List of (label, text) tuples
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            test_ratio: Ratio for test set
            stratify: Whether to stratify by label
            
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        logger.info("Splitting data...")
        
        if stratify:
            # Separate by label
            ham_data = [(label, text) for label, text in data if label == 'ham']
            spam_data = [(label, text) for label, text in data if label == 'spam']
            
            logger.info(f"Ham messages: {len(ham_data)}, Spam messages: {len(spam_data)}")
            
            # Split each class separately
            ham_train_size = int(len(ham_data) * train_ratio)
            ham_val_size = int(len(ham_data) * val_ratio)
            
            spam_train_size = int(len(spam_data) * train_ratio)
            spam_val_size = int(len(spam_data) * val_ratio)
            
            # Combine splits
            train_data = ham_data[:ham_train_size] + spam_data[:spam_train_size]
            val_data = ham_data[ham_train_size:ham_train_size + ham_val_size] + \
                      spam_data[spam_train_size:spam_train_size + spam_val_size]
            test_data = ham_data[ham_train_size + ham_val_size:] + \
                       spam_data[spam_train_size + spam_val_size:]
        else:
            # Random split
            import random
            random.shuffle(data)
            
            train_size = int(len(data) * train_ratio)
            val_size = int(len(data) * val_ratio)
            
            train_data = data[:train_size]
            val_data = data[train_size:train_size + val_size]
            test_data = data[train_size + val_size:]
        
        logger.info(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        return train_data, val_data, test_data
    
    def save_processed_data(self, train_data: List, val_data: List, test_data: List, 
                           vocab: Dict[str, int]) -> None:
        """
        Save processed data and vocabulary.
        
        Args:
            train_data: Training data
            val_data: Validation data
            test_data: Test data
            vocab: Vocabulary dictionary
        """
        logger.info("Saving processed data...")
        
        # Save data splits
        splits = {
            'train': train_data,
            'val': val_data,
            'test': test_data
        }
        
        for split_name, split_data in splits.items():
            file_path = self.processed_data_path / f"{split_name}.csv"
            with open(file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['label', 'text'])
                writer.writerows(split_data)
            logger.info(f"Saved {split_name} data to {file_path}")
        
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
    
    def generate_statistics(self, train_data: List, val_data: List, test_data: List) -> Dict:
        """
        Generate dataset statistics.
        
        Args:
            train_data: Training data
            val_data: Validation data
            test_data: Test data
            
        Returns:
            Dictionary with statistics
        """
        stats = {}
        
        for split_name, split_data in [('train', train_data), ('val', val_data), ('test', test_data)]:
            ham_count = sum(1 for label, _ in split_data if label == 'ham')
            spam_count = sum(1 for label, _ in split_data if label == 'spam')
            
            stats[split_name] = {
                'total': len(split_data),
                'ham': ham_count,
                'spam': spam_count,
                'ham_ratio': ham_count / len(split_data) if split_data else 0,
                'spam_ratio': spam_count / len(split_data) if split_data else 0
            }
        
        return stats
    
    def process(self, min_freq: int = 2, train_ratio: float = 0.7, 
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
        raw_data = self.load_raw_data()
        
        # Preprocess data
        processed_data = self.preprocess_data(raw_data)
        
        # Create vocabulary
        texts = [text for _, text in processed_data]
        vocab = self.create_vocabulary(texts, min_freq)
        
        # Split data
        train_data, val_data, test_data = self.split_data(
            processed_data, train_ratio, val_ratio, test_ratio
        )
        
        # Save processed data
        self.save_processed_data(train_data, val_data, test_data, vocab)
        
        # Generate statistics
        stats = self.generate_statistics(train_data, val_data, test_data)
        
        logger.info("Data processing completed successfully!")
        return stats


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Preprocess SMS spam dataset')
    parser.add_argument('--data-dir', type=str, help='Path to data directory')
    parser.add_argument('--min-freq', type=int, default=2, help='Minimum word frequency')
    parser.add_argument('--train-ratio', type=float, default=0.7, help='Training set ratio')
    parser.add_argument('--val-ratio', type=float, default=0.15, help='Validation set ratio')
    parser.add_argument('--test-ratio', type=float, default=0.15, help='Test set ratio')
    
    args = parser.parse_args()
    
    # Validate ratios
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        logger.error("Ratios must sum to 1.0")
        sys.exit(1)
    
    # Initialize processor
    processor = SMSDataProcessor(args.data_dir)
    
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
        print(f"  Total messages: {split_stats['total']}")
        print(f"  Ham messages: {split_stats['ham']} ({split_stats['ham_ratio']:.2%})")
        print(f"  Spam messages: {split_stats['spam']} ({split_stats['spam_ratio']:.2%})")


if __name__ == "__main__":
    main()
