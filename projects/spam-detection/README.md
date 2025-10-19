# Binary Classification Project - SMS Spam Detection

This project implements three different machine learning approaches for SMS spam detection using the SMS Spam Collection dataset.

## 📊 Dataset Overview

- **Total Messages**: 5,572 SMS messages
- **Ham Messages**: 4,825 (86.6%)
- **Spam Messages**: 747 (13.4%)
- **Class Imbalance**: Moderate (6.5:1 ratio)

## 🚀 Quick Start Commands

### Environment Setup
```bash
# Activate the tfenv environment
source /home/hasib/tfenv/bin/activate

# Navigate to the binary classification project
cd projects/binary
```

### Data Preprocessing
```bash
# Process the raw SMS spam dataset
python scripts/preprocess_data.py

# Optional: Custom preprocessing parameters
python scripts/preprocess_data.py --min-freq 3 --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
```

### Model Training (NEW UNIFIED APPROACH!)

#### Single Algorithm Training
```bash
# Train any available algorithm
python scripts/train_unified.py --algorithm naive_bayes
python scripts/train_unified.py --algorithm logistic_regression
python scripts/train_unified.py --algorithm random_forest
python scripts/train_unified.py --algorithm svm
python scripts/train_unified.py --algorithm knn
python scripts/train_unified.py --algorithm adaboost
python scripts/train_unified.py --algorithm decision_tree

# With hyperparameter tuning
python scripts/train_unified.py --algorithm svm --tuning

# With custom parameters
python scripts/train_unified.py --algorithm naive_bayes --params alpha=0.5 variant=multinomial
```

#### Batch Training (Multiple Algorithms)
```bash
# Train multiple algorithms at once
python scripts/train_batch.py --algorithms naive_bayes logistic_regression random_forest

# Train all available algorithms
python scripts/train_batch.py

# Skip existing models
python scripts/train_batch.py --no-skip
```

#### Available Algorithms
- **naive_bayes**: Multinomial Naive Bayes (fastest, best for text)
- **logistic_regression**: Linear classifier with regularization
- **random_forest**: Ensemble of decision trees
- **svm**: Support Vector Machine with linear kernel
- **knn**: K-Nearest Neighbors
- **adaboost**: Adaptive Boosting ensemble
- **decision_tree**: Single decision tree

### Generate Metrics Report
```bash
# Generate comprehensive metrics report from all trained models
python scripts/generate_metrics_report.py

# Generate metrics for specific model
python scripts/generate_metrics_report.py --model naive_bayes
python scripts/generate_metrics_report.py --model logistic_regression
python scripts/generate_metrics_report.py --model random_forest
```

### Complete Pipeline (NEW UNIFIED APPROACH!)
```bash
# Run the complete pipeline from preprocessing to metrics generation
python scripts/preprocess_data.py

# Train multiple algorithms at once
python scripts/train_batch.py --algorithms naive_bayes logistic_regression random_forest svm knn

# Generate comprehensive metrics report
python scripts/generate_metrics_report.py
```

### Real-time API (NEW!)
```bash
# Start the SMS spam detection API
cd api
pip install -r requirements.txt
./start_api.sh

# Test the API
python demo_api.py

# Access API documentation
# Open http://localhost:8000/docs in your browser
```

## 📁 Project Structure

```
projects/binary/
├── data/
│   ├── raw/
│   │   └── spam.csv                    # Original dataset
│   ├── processed/
│   │   ├── train.csv                   # Training set (3,898 messages)
│   │   ├── val.csv                     # Validation set (835 messages)
│   │   ├── test.csv                    # Test set (837 messages)
│   │   ├── vocabulary.pkl              # Vocabulary (pickle format)
│   │   └── vocabulary.txt              # Vocabulary (text format)
│   └── DATASET_DOCUMENTATION.md        # Dataset documentation
├── scripts/
│   ├── preprocess_data.py              # Data preprocessing script
│   ├── train_unified.py                # Universal model trainer (NEW!)
│   ├── train_batch.py                  # Batch model trainer (NEW!)
│   ├── generate_metrics_report.py      # Comprehensive metrics generation
│   └── legacy/                         # Old training scripts (deprecated)
│       ├── train_naive_bayes.py
│       ├── train_logistic_regression.py
│       └── train_random_forest.py
├── api/                                # Real-time API (NEW!)
│   ├── sms_api.py                      # FastAPI application
│   ├── test_api.py                     # API testing suite
│   ├── demo_api.py                     # API demonstration
│   ├── requirements.txt                 # API dependencies
│   ├── Dockerfile                      # Docker configuration
│   ├── docker-compose.yml              # Docker Compose setup
│   ├── start_api.sh                     # API startup script
│   └── README.md                       # API documentation
├── results/
│   ├── naive_bayes_model.pkl           # Trained Naive Bayes model
│   ├── naive_bayes_vectorizer.pkl      # Naive Bayes vectorizer
│   ├── logistic_regression_model.pkl   # Trained Logistic Regression model
│   ├── logistic_regression_vectorizer.pkl # Logistic Regression vectorizer
│   ├── random_forest_model.pkl         # Trained Random Forest model
│   ├── random_forest_vectorizer.pkl    # Random Forest vectorizer
│   ├── confusion_matrix_*.png          # Confusion matrix plots
│   ├── feature_importance_*.png        # Feature importance plots
│   ├── learning_curve_*.png            # Learning curve plots
│   ├── model_comparison_charts.png     # Model comparison visualization
│   └── detailed_metrics_report.txt     # Comprehensive metrics report
└── README.md                           # This file
```

## 🎯 Model Performance Summary

### Naive Bayes (Multinomial)
- **Test Accuracy**: 98.69%
- **Test Precision**: 96.36%
- **Test Recall**: 93.81%
- **Test F1-Score**: 95.07%
- **Test ROC-AUC**: 99.11%
- **Cross-validation**: 98.35% ± 0.58%

### Logistic Regression
- **Test Accuracy**: 97.37%
- **Test Precision**: 94.17%
- **Test Recall**: 85.84%
- **Test F1-Score**: 89.81%
- **Test ROC-AUC**: 99.18%

### Random Forest
- **Test Accuracy**: 97.73%
- **Test Precision**: 100.00%
- **Test Recall**: 83.19%
- **Test F1-Score**: 90.82%
- **Test ROC-AUC**: 99.79%
- **Cross-validation**: 97.80% ± 0.49%

## 🔧 Script Parameters

### Preprocessing Script (`preprocess_data.py`)
- `--data-dir`: Path to data directory
- `--min-freq`: Minimum word frequency for vocabulary (default: 2)
- `--train-ratio`: Training set ratio (default: 0.7)
- `--val-ratio`: Validation set ratio (default: 0.15)
- `--test-ratio`: Test set ratio (default: 0.15)

### Training Scripts
- `--data-dir`: Path to data directory
- `--results-dir`: Path to results directory
- `--vectorizer`: Type of vectorizer ('count' or 'tfidf')
- `--no-tuning`: Skip hyperparameter tuning

### Naive Bayes Specific
- `--alpha`: Smoothing parameter (default: 1.0)
- `--variant`: Naive Bayes variant ('multinomial' or 'bernoulli')

## 📈 Key Features

### Data Preprocessing
- Text cleaning and normalization
- URL, email, and phone number replacement
- Vocabulary creation with frequency filtering
- Stratified train/validation/test splits

### Model Training
- Comprehensive evaluation metrics
- Cross-validation
- Confusion matrix visualization
- Feature importance analysis
- Model persistence (pickle files)

### Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC score
- Confusion matrix analysis
- Cross-validation scores
- Feature importance rankings

## 🎨 Visualizations Generated

1. **Confusion Matrices**: Validation and test set confusion matrices
2. **Feature Importance**: Top features for each model
3. **Learning Curves**: Training progress (Random Forest)
4. **Metrics Report**: Comprehensive performance comparison

## 💡 Recommendations

1. **Start with Naive Bayes**: Best performance for text classification
2. **Use TF-IDF for Linear Models**: Logistic Regression and Random Forest
3. **Use Count Vectorization for Naive Bayes**: More appropriate for probabilistic models
4. **Enable Hyperparameter Tuning**: For production models (slower but better performance)

## 🐛 Troubleshooting

### Common Issues
1. **ModuleNotFoundError**: Make sure tfenv environment is activated
2. **FileNotFoundError**: Run preprocessing script first
3. **Memory Issues**: Reduce vocabulary size or use smaller datasets

### Environment Issues
```bash
# Check if environment is activated
which python
# Should show: /home/hasib/tfenv/bin/python

# Install missing packages
pip install numpy pandas scikit-learn matplotlib seaborn
```

## 📚 References

- SMS Spam Collection v.1 Dataset
- Scikit-learn Documentation
- Machine Learning Best Practices for Text Classification

---

*Last updated: October 19, 2025*
