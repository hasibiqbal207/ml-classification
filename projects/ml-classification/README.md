# ML Classification Project

A comprehensive machine learning classification project that demonstrates multiple algorithms implemented across different frameworks (Scikit-learn, TensorFlow, PyTorch).

## Project Structure

```
projects/ml-classification/
├── src/
│   ├── pipelines/                # Training and evaluation pipelines
│   │   ├── train.py
│   │   └── evaluate.py
│   └── utils/                    # Project-specific utilities
│       ├── trainer_tf.py
│       └── trainer_pt.py
├── notebooks/                    # Jupyter notebooks for exploration
│   ├── 01_data_exploration.ipynb
│   └── 02_model_comparison.ipynb
├── data/                         # Data directory
│   ├── raw/                      # Raw datasets
│   ├── processed/                 # Processed datasets
│   └── external_datasets.txt     # External dataset references
├── reports/                      # Generated reports and visualizations
│   ├── figures/
│   ├── metrics/
│   └── logs/
├── scripts/                      # Utility scripts
│   ├── download_data.py
│   └── preprocess_data.py
├── config.yaml                   # Project configuration
├── requirements.txt              # Project dependencies
└── README.md                    # This file
```

## Features

### Algorithms Implemented
- **Logistic Regression**: Scikit-learn, TensorFlow, PyTorch
- **Decision Tree**: Scikit-learn, PyTorch
- **CNN**: TensorFlow, PyTorch
- **LSTM**: TensorFlow, PyTorch
- **K-Means**: Scikit-learn

### Key Features
- **Multiple Framework Support**: Compare implementations across different ML frameworks
- **Comprehensive Evaluation**: Built-in metrics, visualization, and reporting
- **Configurable Pipeline**: YAML-based configuration system
- **Data Management**: Automated data loading, preprocessing, and management
- **Visualization**: Rich plotting and visualization utilities
- **Cross-Validation**: Built-in cross-validation support
- **Model Persistence**: Save and load trained models

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Data
```bash
python scripts/download_data.py --dataset iris
```

### 3. Preprocess Data
```bash
python scripts/preprocess_data.py --dataset iris
```

### 4. Train Model
```bash
python pipelines/train.py --algorithm logistic_regression --implementation sklearn --dataset iris --evaluate --visualize
```

### 5. Evaluate Model
```bash
python pipelines/evaluate.py --algorithm logistic_regression --implementation sklearn --model_path models/logistic_regression_sklearn_final.h5 --dataset iris --visualize --report
```

## Usage Examples

### Training Multiple Models
```python
from pipelines.train import MLPipeline

# Initialize pipeline
pipeline = MLPipeline("config.yaml")

# Load and preprocess data
pipeline.load_data(dataset_name="iris")
pipeline.preprocess_data()

# Train different models
pipeline.train_model("logistic_regression", "sklearn")
pipeline.train_model("logistic_regression", "tensorflow")
pipeline.train_model("decision_tree", "sklearn")

# Evaluate and compare
for model_name in pipeline.models.keys():
    pipeline.evaluate_model(model_name)

# Compare results
comparison = pipeline.compare_models("accuracy")
print(comparison)
```

### Custom Configuration
```yaml
# config.yaml
data:
  test_size: 0.3
  preprocessing:
    handle_missing: true
    scale_features: true

models:
  logistic_regression:
    sklearn:
      max_iter: 2000
      random_state: 42
    tensorflow:
      learning_rate: 0.01
      epochs: 150

training:
  validation_split: 0.2
  early_stopping:
    patience: 15
```

## Configuration

The project uses YAML configuration files for easy customization. Key configuration sections:

- **data**: Data loading and preprocessing settings
- **models**: Algorithm-specific parameters
- **training**: Training hyperparameters
- **evaluation**: Evaluation metrics and settings
- **paths**: File and directory paths

## Available Datasets

- **iris**: Classic iris flower dataset
- **wine**: Wine quality dataset
- **breast_cancer**: Breast cancer classification dataset

## Algorithm Comparison

This project allows you to compare the same algorithm implemented in different frameworks:

```python
# Compare logistic regression implementations
pipeline.train_model("logistic_regression", "sklearn")
pipeline.train_model("logistic_regression", "tensorflow")
pipeline.train_model("logistic_regression", "pytorch")

# Compare results
comparison = pipeline.compare_models("accuracy")
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
