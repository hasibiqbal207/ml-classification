#!/usr/bin/env python3
"""
Batch Model Training Script for 20 Newsgroups Classification

This script trains multiple algorithms using the unified training framework.
"""

import argparse
import time
from pathlib import Path
from typing import List, Dict, Any
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
import sys
sys.path.append(str(project_root))

# Import the unified trainer
from train_unified import UniversalModelTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BatchTrainer:
    """
    Batch trainer for multiple algorithms.
    """
    
    def __init__(self, data_dir: str = None, results_dir: str = None):
        """
        Initialize batch trainer.
        
        Args:
            data_dir: Path to data directory
            results_dir: Path to results directory
        """
        self.data_dir = data_dir
        self.results_dir = results_dir
        
        # Default algorithms to train
        self.default_algorithms = [
            'naive_bayes',
            'logistic_regression', 
            'random_forest',
            'svm',
            'knn',
            'adaboost'
        ]
        
        # Algorithm-specific parameters
        self.algorithm_params = {
            'naive_bayes': {'variant': 'multinomial', 'alpha': 1.0},
            'logistic_regression': {'random_state': 42, 'max_iter': 1000},
            'random_forest': {'n_estimators': 100, 'random_state': 42},
            'svm': {'kernel': 'linear', 'random_state': 42},
            'knn': {'n_neighbors': 5},
            'adaboost': {'n_estimators': 50, 'random_state': 42},
            'decision_tree': {'random_state': 42}
        }
    
    def train_multiple_algorithms(self, algorithms: List[str] = None, 
                                 hyperparameter_tuning: bool = False,
                                 skip_existing: bool = True) -> Dict[str, Any]:
        """
        Train multiple algorithms.
        
        Args:
            algorithms: List of algorithms to train (default: all available)
            hyperparameter_tuning: Whether to perform hyperparameter tuning
            skip_existing: Whether to skip algorithms that already have trained models
            
        Returns:
            dict: Results for all trained algorithms
        """
        if algorithms is None:
            algorithms = self.default_algorithms
        
        logger.info(f"Starting batch training for algorithms: {algorithms}")
        
        results = {}
        start_time = time.time()
        
        for i, algorithm in enumerate(algorithms, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Training {algorithm} ({i}/{len(algorithms)})")
            logger.info(f"{'='*60}")
            
            # Check if model already exists
            if skip_existing:
                model_path = Path(self.results_dir) / "models" / f"{algorithm}_model.pkl"
                if model_path.exists():
                    logger.info(f"Skipping {algorithm} - model already exists")
                    continue
            
            try:
                # Create trainer
                trainer = UniversalModelTrainer(self.data_dir, self.results_dir)
                
                # Get algorithm-specific parameters
                params = self.algorithm_params.get(algorithm, {})
                
                # Train algorithm
                algorithm_start = time.time()
                result = trainer.train_and_evaluate(
                    algorithm_name=algorithm,
                    hyperparameter_tuning=hyperparameter_tuning,
                    **params
                )
                algorithm_time = time.time() - algorithm_start
                
                results[algorithm] = {
                    **result,
                    'training_time_seconds': algorithm_time
                }
                
                logger.info(f"✅ {algorithm} completed in {algorithm_time:.2f} seconds")
                
            except Exception as e:
                logger.error(f"❌ {algorithm} failed: {e}")
                results[algorithm] = {
                    'error': str(e),
                    'status': 'failed'
                }
        
        total_time = time.time() - start_time
        
        # Print summary
        self.print_summary(results, total_time)
        
        return results
    
    def print_summary(self, results: Dict[str, Any], total_time: float) -> None:
        """
        Print training summary.
        
        Args:
            results: Results from all algorithms
            total_time: Total training time
        """
        print(f"\n{'='*80}")
        print("BATCH TRAINING SUMMARY")
        print(f"{'='*80}")
        
        successful = []
        failed = []
        
        for algorithm, result in results.items():
            if 'error' in result:
                failed.append(algorithm)
            else:
                successful.append(algorithm)
        
        print(f"Total algorithms: {len(results)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")
        print(f"Total time: {total_time:.2f} seconds")
        
        if successful:
            print(f"\n✅ SUCCESSFUL ALGORITHMS:")
            for algorithm in successful:
                result = results[algorithm]
                if 'test' in result:
                    test_metrics = result['test']
                    accuracy = test_metrics.get('accuracy', 'N/A')
                    print(f"  {algorithm}: Accuracy = {accuracy:.4f}")
                else:
                    print(f"  {algorithm}: Completed")
        
        if failed:
            print(f"\n❌ FAILED ALGORITHMS:")
            for algorithm in failed:
                error = results[algorithm]['error']
                print(f"  {algorithm}: {error}")
        
        print(f"\n{'='*80}")


def main():
    """Main function to run batch training."""
    parser = argparse.ArgumentParser(description='Batch Model Trainer for 20 Newsgroups Classification')
    parser.add_argument('--algorithms', nargs='*', 
                       help='Algorithms to train (default: all available)')
    parser.add_argument('--data-dir', help='Path to data directory')
    parser.add_argument('--results-dir', help='Path to results directory')
    parser.add_argument('--tuning', action='store_true', help='Perform hyperparameter tuning')
    parser.add_argument('--no-skip', action='store_true', help='Don\'t skip existing models')
    
    args = parser.parse_args()
    
    # Create batch trainer
    batch_trainer = BatchTrainer(args.data_dir, args.results_dir)
    
    # Train algorithms
    try:
        results = batch_trainer.train_multiple_algorithms(
            algorithms=args.algorithms,
            hyperparameter_tuning=args.tuning,
            skip_existing=not args.no_skip
        )
        
        print(f"\n🎉 Batch training completed!")
        print(f"Check the results directory for trained models and visualizations.")
        
    except Exception as e:
        logger.error(f"Batch training failed: {e}")
        print(f"❌ Batch training failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
