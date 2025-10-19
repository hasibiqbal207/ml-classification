# Multilabel Classification Project - GoEmotions Analysis

This project implements multiple machine learning approaches for emotion analysis using the GoEmotions dataset from Google Research.

## 📊 Dataset Overview

- **Total Comments**: 205,862 Reddit comments (after preprocessing)
- **Emotion Categories**: 27 different emotions
- **Problem Type**: Multilabel Classification (multiple emotions per comment)
- **Average Labels per Comment**: 1.19 (sparse multilabel)

### Emotion Categories

The dataset includes 27 emotion labels organized into three main groups:

**Positive Emotions:**
- `admiration`, `amusement`, `approval`, `caring`, `excitement`, `gratitude`, `joy`, `love`, `optimism`, `pride`, `relief`

**Negative Emotions:**
- `anger`, `annoyance`, `disappointment`, `disapproval`, `disgust`, `embarrassment`, `fear`, `grief`, `nervousness`, `remorse`, `sadness`

**Neutral/Other:**
- `confusion`, `curiosity`, `desire`, `realization`, `surprise`, `neutral`

### Label Distribution
- **Most common**: `neutral` (25.94% of comments)
- **Least common**: `grief` (0.32%), `pride` (0.62%), `relief` (0.62%)
- **Sparse labels**: Most comments have 0-2 emotion labels

## 🚀 Quick Start Commands

### Environment Setup
```bash
# Activate the tfenv environment
source /home/hasib/tfenv/bin/activate

# Navigate to the emotion analysis project
cd projects/emotion-analysis
```

### Data Preprocessing
```bash
# Process the raw GoEmotions dataset
python scripts/preprocess_data.py

# Optional: Custom preprocessing parameters
python scripts/preprocess_data.py --min-freq 3 --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15
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

### Real-time API (Coming Soon!)
```bash
# Start the emotion analysis API
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
projects/emotion-analysis/
├── data/
│   ├── raw/
│   │   └── go_emotions_dataset.csv     # Original dataset (211,225 comments)
│   ├── processed/
│   │   ├── train_texts.csv             # Training texts (144,103 samples)
│   │   ├── val_texts.csv               # Validation texts (30,879 samples)
│   │   ├── test_texts.csv              # Test texts (30,880 samples)
│   │   ├── train_labels.csv            # Training labels (binary matrix)
│   │   ├── val_labels.csv              # Validation labels (binary matrix)
│   │   ├── test_labels.csv             # Test labels (binary matrix)
│   │   ├── vocabulary.pkl              # Vocabulary (30,125 words)
│   │   ├── vocabulary.txt              # Vocabulary (text format)
│   │   └── emotion_categories.pkl      # Emotion categories list
│   └── DATASET_DOCUMENTATION.md        # Dataset documentation
├── scripts/
│   ├── preprocess_data.py               # Data preprocessing script
│   ├── train_unified.py                 # Universal model trainer
│   ├── train_batch.py                   # Batch model trainer
│   └── generate_metrics_report.py       # Comprehensive metrics generation
├── api/                                 # Real-time API (Coming Soon!)
│   ├── emotion_api.py                   # FastAPI application
│   ├── demo_api.py                      # API demonstration
│   ├── test_api.py                      # API testing suite
│   ├── requirements.txt                 # API dependencies
│   ├── start_api.sh                     # API startup script
│   └── README.md                        # API documentation
├── results/
│   ├── models/
│   │   ├── naive_bayes_model.pkl        # Trained Naive Bayes model
│   │   ├── naive_bayes_vectorizer.pkl   # Naive Bayes vectorizer
│   │   ├── naive_bayes_emotion_categories.pkl # Naive Bayes emotion categories
│   │   ├── logistic_regression_model.pkl # Trained Logistic Regression model
│   │   ├── logistic_regression_vectorizer.pkl # Logistic Regression vectorizer
│   │   ├── logistic_regression_emotion_categories.pkl # Logistic Regression emotion categories
│   │   └── ...                          # Other model files
│   ├── visualizations/
│   │   ├── model_comparison_charts.png  # Model comparison visualization
│   │   ├── emotion_performance_heatmap.png # Per-emotion performance heatmap
│   │   └── ...                          # Other visualizations
│   └── reports/
│       └── detailed_metrics_report.txt  # Comprehensive metrics report
└── README.md                            # This file
```

## 🎯 Expected Model Performance

### Typical Performance Ranges (Multilabel Classification)
- **Accuracy**: 60-85% (exact match accuracy)
- **F1-Score (Macro)**: 0.3-0.6 (average across emotions)
- **F1-Score (Micro)**: 0.4-0.7 (overall performance)
- **Hamming Loss**: 0.1-0.3 (lower is better)
- **Jaccard Score**: 0.2-0.5 (intersection over union)

*Note: Multilabel classification is inherently more challenging than binary/multiclass classification. Actual performance may vary based on preprocessing parameters and hyperparameter tuning.*

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
- Reddit-specific pattern handling ([NAME], u/username, r/subreddit)
- Multilabel data preparation
- Vocabulary creation with frequency filtering
- Stratified train/validation/test splits

### Model Training
- Multilabel classification support
- Comprehensive multilabel evaluation metrics
- Per-emotion performance analysis
- Model persistence (pickle files)
- Emotion category handling

### Evaluation Metrics
- **Accuracy**: Exact match accuracy
- **Hamming Loss**: Fraction of wrong labels
- **Jaccard Score**: Intersection over union
- **Precision/Recall/F1**: Macro and micro averages
- **Per-Emotion Metrics**: Individual emotion performance

## 🎨 Visualizations Generated

1. **Model Comparison Charts**: Accuracy, F1-Scores, Hamming Loss, Jaccard Score
2. **Emotion Performance Heatmap**: Per-emotion F1-scores across algorithms
3. **Metrics Report**: Comprehensive performance analysis

## 💡 Recommendations

1. **Start with Naive Bayes**: Good baseline for multilabel text classification
2. **Use TF-IDF for Linear Models**: Logistic Regression and Random Forest
3. **Use Count Vectorization for Naive Bayes**: More appropriate for probabilistic models
4. **Consider Class Imbalance**: Some emotions are very rare (grief, pride, relief)
5. **Enable Hyperparameter Tuning**: For production models (slower but better performance)

## 🔍 Multilabel Classification Challenges

### Sparse Labels
- Most comments have 0-2 emotion labels
- Average of 1.19 labels per comment
- Some emotions are extremely rare

### Class Imbalance
- `neutral` appears in 25.94% of comments
- `grief` appears in only 0.32% of comments
- Consider threshold tuning for rare emotions

### Evaluation Complexity
- Multiple metrics needed for comprehensive evaluation
- Per-emotion analysis important for understanding model behavior
- Consider both macro and micro averages

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
pip install numpy pandas scikit-learn matplotlib seaborn
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

## 📊 Dataset Statistics

### Preprocessing Results
- **Original Comments**: 211,225
- **Processed Comments**: 205,862 (filtered out very short/empty)
- **Vocabulary Size**: 30,125 words (min_freq=3)
- **Train Set**: 144,103 samples (70%)
- **Validation Set**: 30,879 samples (15%)
- **Test Set**: 30,880 samples (15%)

### Label Distribution (Top 10)
1. `neutral`: 53,404 (25.94%)
2. `approval`: 17,377 (8.44%)
3. `admiration`: 16,622 (8.07%)
4. `annoyance`: 13,429 (6.52%)
5. `disapproval`: 11,354 (5.52%)
6. `gratitude`: 11,018 (5.35%)
7. `curiosity`: 9,542 (4.64%)
8. `optimism`: 8,670 (4.21%)
9. `realization`: 8,770 (4.26%)
10. `disappointment`: 8,401 (4.08%)

## 📚 References

- GoEmotions Dataset (Google Research)
- Scikit-learn Documentation
- Multilabel Classification Best Practices
- Reddit Comment Analysis

---

*Last updated: October 19, 2025*
