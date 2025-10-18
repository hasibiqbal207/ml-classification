"""
Configuration loader utilities.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Union


class ConfigLoader:
    """
    Utility class for loading and managing configuration files.
    """
    
    def __init__(self, config_path="config.yaml"):
        """
        Initialize ConfigLoader.
        
        Args:
            config_path (str): Path to configuration file
        """
        self.config_path = Path(config_path)
        self.config = {}
        
    def load_config(self, config_path=None):
        """
        Load configuration from file.
        
        Args:
            config_path (str): Path to configuration file
            
        Returns:
            dict: Configuration dictionary
        """
        if config_path:
            self.config_path = Path(config_path)
            
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
        if self.config_path.suffix.lower() == '.yaml' or self.config_path.suffix.lower() == '.yml':
            with open(self.config_path, 'r') as file:
                self.config = yaml.safe_load(file)
        elif self.config_path.suffix.lower() == '.json':
            with open(self.config_path, 'r') as file:
                self.config = json.load(file)
        else:
            raise ValueError("Configuration file must be .yaml, .yml, or .json")
            
        return self.config
        
    def save_config(self, config=None, config_path=None):
        """
        Save configuration to file.
        
        Args:
            config (dict): Configuration dictionary
            config_path (str): Path to save configuration file
        """
        if config:
            self.config = config
            
        if config_path:
            self.config_path = Path(config_path)
            
        if self.config_path.suffix.lower() == '.yaml' or self.config_path.suffix.lower() == '.yml':
            with open(self.config_path, 'w') as file:
                yaml.dump(self.config, file, default_flow_style=False)
        elif self.config_path.suffix.lower() == '.json':
            with open(self.config_path, 'w') as file:
                json.dump(self.config, file, indent=4)
        else:
            raise ValueError("Configuration file must be .yaml, .yml, or .json")
            
    def get(self, key, default=None):
        """
        Get configuration value by key.
        
        Args:
            key (str): Configuration key (supports dot notation)
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
            
    def set(self, key, value):
        """
        Set configuration value by key.
        
        Args:
            key (str): Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
            
        config[keys[-1]] = value
        
    def update(self, updates):
        """
        Update configuration with new values.
        
        Args:
            updates (dict): Dictionary of updates
        """
        self._update_nested(self.config, updates)
        
    def _update_nested(self, config, updates):
        """
        Recursively update nested dictionary.
        
        Args:
            config (dict): Configuration dictionary
            updates (dict): Updates to apply
        """
        for key, value in updates.items():
            if isinstance(value, dict) and key in config and isinstance(config[key], dict):
                self._update_nested(config[key], value)
            else:
                config[key] = value
                
    def get_model_config(self, model_name):
        """
        Get configuration for a specific model.
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            dict: Model configuration
        """
        return self.get(f'models.{model_name}', {})
        
    def get_data_config(self):
        """
        Get data configuration.
        
        Returns:
            dict: Data configuration
        """
        return self.get('data', {})
        
    def get_training_config(self):
        """
        Get training configuration.
        
        Returns:
            dict: Training configuration
        """
        return self.get('training', {})
        
    def get_evaluation_config(self):
        """
        Get evaluation configuration.
        
        Returns:
            dict: Evaluation configuration
        """
        return self.get('evaluation', {})


def create_default_config():
    """
    Create a default configuration file.
    
    Returns:
        dict: Default configuration
    """
    config = {
        'data': {
            'raw_path': 'data/raw',
            'processed_path': 'data/processed',
            'test_size': 0.2,
            'random_state': 42,
            'preprocessing': {
                'handle_missing': True,
                'encode_categorical': True,
                'scale_features': True
            }
        },
        'models': {
            'logistic_regression': {
                'sklearn': {
                    'random_state': 42,
                    'max_iter': 1000
                },
                'tensorflow': {
                    'learning_rate': 0.01,
                    'epochs': 100,
                    'batch_size': 32
                },
                'pytorch': {
                    'learning_rate': 0.01,
                    'epochs': 100,
                    'batch_size': 32
                }
            },
            'decision_tree': {
                'sklearn': {
                    'random_state': 42,
                    'max_depth': None,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1,
                    'criterion': 'gini'
                },
                'pytorch': {
                    'learning_rate': 0.01,
                    'epochs': 100,
                    'batch_size': 32
                }
            },
            'cnn': {
                'tensorflow': {
                    'learning_rate': 0.001,
                    'epochs': 100,
                    'batch_size': 32,
                    'lstm_units': 128
                },
                'pytorch': {
                    'learning_rate': 0.001,
                    'epochs': 100,
                    'batch_size': 32,
                    'hidden_size': 128
                }
            },
            'lstm': {
                'tensorflow': {
                    'learning_rate': 0.001,
                    'epochs': 100,
                    'batch_size': 32,
                    'lstm_units': 128
                },
                'pytorch': {
                    'learning_rate': 0.001,
                    'epochs': 100,
                    'batch_size': 32,
                    'hidden_size': 128
                }
            },
            'kmeans': {
                'sklearn': {
                    'n_clusters': 8,
                    'random_state': 42,
                    'max_iter': 300,
                    'init': 'k-means++',
                    'n_init': 10
                }
            }
        },
        'training': {
            'validation_split': 0.2,
            'early_stopping': {
                'patience': 10,
                'monitor': 'val_loss'
            },
            'callbacks': {
                'reduce_lr': {
                    'factor': 0.5,
                    'patience': 5,
                    'min_lr': 1e-7
                }
            }
        },
        'evaluation': {
            'metrics': ['accuracy', 'precision', 'recall', 'f1_score'],
            'cross_validation': {
                'cv': 5,
                'scoring': 'accuracy'
            }
        },
        'paths': {
            'models': 'models',
            'reports': 'reports',
            'logs': 'logs'
        }
    }
    
    return config


def load_config(config_path="config.yaml"):
    """
    Load configuration from file.
    
    Args:
        config_path (str): Path to configuration file
        
    Returns:
        ConfigLoader: Configuration loader instance
    """
    loader = ConfigLoader(config_path)
    loader.load_config()
    return loader
