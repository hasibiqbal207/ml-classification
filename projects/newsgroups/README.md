# Multiclass Classification Project - 20 Newsgroups

This project implements multiple machine learning approaches for 20 newsgroups classification using the classic 20 Newsgroups dataset.

## 📊 Dataset Overview

- **Total Categories**: 20 newsgroups
- **Total Documents**: ~20,000 news articles
- **Average Documents per Category**: ~1,000
- **Problem Type**: Multiclass Classification (20 classes)

### Categories

The dataset includes 20 newsgroup categories organized into 6 main groups:

**Computer Technology:**
- `comp.graphics`
- `comp.os.ms-windows.misc`
- `comp.sys.ibm.pc.hardware`
- `comp.sys.mac.hardware`
- `comp.windows.x`

**Recreation:**
- `rec.autos`
- `rec.motorcycles`
- `rec.sport.baseball`
- `rec.sport.hockey`

**Science:**
- `sci.crypt`
- `sci.electronics`
- `sci.med`
- `sci.space`

**Religion:**
- `alt.atheism`
- `soc.religion.christian`
- `talk.religion.misc`

**Politics:**
- `talk.politics.guns`
- `talk.politics.mideast`
- `talk.politics.misc`

**Miscellaneous:**
- `misc.forsale`

## 🚀 Quick Start Commands

### Environment Setup
```bash
# Activate the tfenv environment
source /home/hasib/tfenv/bin/activate

# Navigate to the newsgroups project
cd projects/newsgroups
```

### Data Preprocessing
```bash
# Process the raw 20 newsgroups dataset
python scripts/preprocess_data.py

# Optional: Custom preprocessing parameters
python scripts/preprocess_data.py --min-freq 5 --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15
```

### Model Training (UNIFIED APPROACH!)

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
- **naive_bayes**: Multinomial Naive Bayes (fastest, good for text)
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

### Complete Pipeline (UNIFIED APPROACH!)
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
# Start the 20 newsgroups classification API
cd api
pip install -r requirements.txt
./start_api.sh

# Test the API
python demo_api.py

# Run comprehensive tests
python test_api.py

# Access API documentation
# Open http://localhost:8000/docs in your browser
```

## 📁 Project Structure

```
projects/newsgroups/
├── data/
│   ├── raw/
│   │   ├── alt.atheism.txt
│   │   ├── comp.graphics.txt
│   │   ├── comp.os.ms-windows.misc.txt
│   │   ├── comp.sys.ibm.pc.hardware.txt
│   │   ├── comp.sys.mac.hardware.txt
│   │   ├── comp.windows.x.txt
│   │   ├── misc.forsale.txt
│   │   ├── rec.autos.txt
│   │   ├── rec.motorcycles.txt
│   │   ├── rec.sport.baseball.txt
│   │   ├── rec.sport.hockey.txt
│   │   ├── sci.crypt.txt
│   │   ├── sci.electronics.txt
│   │   ├── sci.med.txt
│   │   ├── sci.space.txt
│   │   ├── soc.religion.christian.txt
│   │   ├── talk.politics.guns.txt
│   │   ├── talk.politics.mideast.txt
│   │   ├── talk.politics.misc.txt
│   │   ├── talk.religion.misc.txt
│   │   └── list.csv                    # Document mapping
│   ├── processed/
│   │   ├── train.csv                   # Training set
│   │   ├── val.csv                      # Validation set
│   │   ├── test.csv                     # Test set
│   │   ├── vocabulary.pkl               # Vocabulary (pickle format)
│   │   └── vocabulary.txt               # Vocabulary (text format)
│   └── DATASET_DOCUMENTATION.md         # Dataset documentation
├── scripts/
│   ├── preprocess_data.py               # Data preprocessing script
│   ├── train_unified.py                 # Universal model trainer
│   ├── train_batch.py                   # Batch model trainer
│   └── generate_metrics_report.py       # Comprehensive metrics generation
├── api/                                 # Real-time API
│   ├── newsgroups_api.py                # FastAPI application
│   ├── test_api.py                      # API testing suite
│   ├── demo_api.py                      # API demonstration
│   ├── requirements.txt                 # API dependencies
│   ├── start_api.sh                     # API startup script
│   └── README.md                        # API documentation
├── results/
│   ├── models/
│   │   ├── naive_bayes_model.pkl        # Trained Naive Bayes model
│   │   ├── naive_bayes_vectorizer.pkl   # Naive Bayes vectorizer
│   │   ├── naive_bayes_label_encoder.pkl # Naive Bayes label encoder
│   │   ├── logistic_regression_model.pkl # Trained Logistic Regression model
│   │   ├── logistic_regression_vectorizer.pkl # Logistic Regression vectorizer
│   │   ├── logistic_regression_label_encoder.pkl # Logistic Regression label encoder
│   │   ├── random_forest_model.pkl      # Trained Random Forest model
│   │   ├── random_forest_vectorizer.pkl  # Random Forest vectorizer
│   │   ├── random_forest_label_encoder.pkl # Random Forest label encoder
│   │   └── ...                          # Other model files
│   ├── visualizations/
│   │   ├── confusion_matrix_*.png        # Confusion matrix plots
│   │   ├── feature_importance_*.png      # Feature importance plots
│   │   ├── learning_curve_*.png         # Learning curve plots
│   │   └── model_comparison_charts.png  # Model comparison visualization
│   └── reports/
│       └── detailed_metrics_report.txt  # Comprehensive metrics report
└── README.md                            # This file
```

## 🎯 Expected Model Performance

### Typical Performance Ranges
- **Naive Bayes**: 70-85% accuracy
- **Logistic Regression**: 75-90% accuracy
- **Random Forest**: 80-95% accuracy
- **SVM**: 85-95% accuracy
- **KNN**: 60-80% accuracy
- **AdaBoost**: 70-85% accuracy
- **Decision Tree**: 60-80% accuracy

*Note: Actual performance may vary based on preprocessing parameters and hyperparameter tuning.*

## 🔧 Script Parameters

### Preprocessing Script (`preprocess_data.py`)
- `--data-dir`: Path to data directory
- `--min-freq`: Minimum word frequency for vocabulary (default: 5)
- `--train-ratio`: Training set ratio (default: 0.7)
- `--val-ratio`: Validation set ratio (default: 0.15)
- `--test-ratio`: Test set ratio (default: 0.15)

### Training Scripts
- `--data-dir`: Path to data directory
- `--results-dir`: Path to results directory
- `--tuning`: Perform hyperparameter tuning
- `--params`: Additional parameters as key=value pairs

### Naive Bayes Specific
- `--alpha`: Smoothing parameter (default: 1.0)
- `--variant`: Naive Bayes variant ('multinomial' or 'bernoulli')

## 📈 Key Features

### Data Preprocessing
- Text cleaning and normalization
- URL and email replacement
- Header removal from newsgroup posts
- Document splitting and parsing
- Vocabulary creation with frequency filtering
- Stratified train/validation/test splits

### Model Training
- Comprehensive evaluation metrics
- Cross-validation support
- Confusion matrix visualization
- Feature importance analysis
- Model persistence (pickle files)
- Label encoding for multiclass classification

### Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score
- Macro and weighted averages
- Per-class performance metrics
- Confusion matrix analysis
- Model comparison visualizations

## 🎨 Visualizations Generated

1. **Confusion Matrices**: Validation and test set confusion matrices
2. **Feature Importance**: Top features for each model
3. **Learning Curves**: Training progress (where applicable)
4. **Model Comparison Charts**: Performance comparison across algorithms
5. **Metrics Report**: Comprehensive performance analysis

## 💡 Recommendations

1. **Start with Naive Bayes**: Good baseline for text classification
2. **Use TF-IDF for Linear Models**: Logistic Regression and Random Forest
3. **Use Count Vectorization for Naive Bayes**: More appropriate for probabilistic models
4. **Enable Hyperparameter Tuning**: For production models (slower but better performance)
5. **Consider Class Imbalance**: Some categories may have fewer samples

## 🔌 API Usage

### Basic Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "Looking for advice on computer graphics programming"}'
```

### Batch Prediction
```bash
curl -X POST "http://localhost:8000/predict/batch" \
     -H "Content-Type: application/json" \
     -d '{"texts": ["Text 1", "Text 2", "Text 3"]}'
```

### Model Management
```bash
# Load a specific model
curl -X POST "http://localhost:8000/models/naive_bayes/load"

# List available models
curl -X GET "http://localhost:8000/models"

# Health check
curl -X GET "http://localhost:8000/health"
```

## 🐛 Troubleshooting

### Common Issues
1. **ModuleNotFoundError**: Make sure tfenv environment is activated
2. **FileNotFoundError**: Run preprocessing script first
3. **Memory Issues**: Reduce vocabulary size or use smaller datasets
4. **API Connection Error**: Ensure API server is running

### Environment Issues
```bash
# Check if environment is activated
which python
# Should show: /home/hasib/tfenv/bin/python

# Install missing packages
pip install numpy pandas scikit-learn matplotlib seaborn fastapi uvicorn
```

### Data Issues
```bash
# Check if raw data exists
ls data/raw/

# Check if processed data exists
ls data/processed/

# Re-run preprocessing if needed
python scripts/preprocess_data.py
```

## 📚 References

- 20 Newsgroups Dataset
- Scikit-learn Documentation
- FastAPI Documentation
- Machine Learning Best Practices for Text Classification

---

*Last updated: October 19, 2025*
