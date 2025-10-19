# Results Directory Organization

This directory contains the organized results from the SMS spam classification project.

## Directory Structure

```
results/
├── models/                    # All trained models and vectorizers
│   ├── naive_bayes_model.pkl
│   ├── naive_bayes_vectorizer.pkl
│   ├── logistic_regression_model.pkl
│   ├── logistic_regression_vectorizer.pkl
│   ├── random_forest_model.pkl
│   ├── random_forest_vectorizer.pkl
│   ├── svm_model.pkl
│   ├── svm_vectorizer.pkl
│   ├── knn_model.pkl
│   └── knn_vectorizer.pkl
├── visualizations/           # All charts, plots, and visualizations
│   ├── model_comparison_charts.png
│   ├── naive_bayes_test_confusion_matrix.png
│   ├── naive_bayes_validation_confusion_matrix.png
│   ├── logistic_regression_feature_importance.png
│   ├── logistic_regression_test_confusion_matrix.png
│   ├── logistic_regression_validation_confusion_matrix.png
│   ├── random_forest_feature_importance.png
│   ├── random_forest_learning_curve.png
│   ├── random_forest_test_confusion_matrix.png
│   └── random_forest_validation_confusion_matrix.png
└── reports/                  # All text reports and metrics
    └── detailed_metrics_report.txt
```

## File Types

### Models (`models/`)
- **Model files** (`*_model.pkl`): Trained machine learning models
- **Vectorizer files** (`*_vectorizer.pkl`): Text preprocessing and feature extraction objects

### Visualizations (`visualizations/`)
- **Confusion matrices**: Test and validation confusion matrices for each model
- **Feature importance plots**: Shows which features are most important for predictions
- **Learning curves**: Shows model performance vs training data size
- **Model comparison charts**: Side-by-side comparison of all models

### Reports (`reports/`)
- **Metrics reports**: Detailed performance metrics and statistics
- **Analysis summaries**: Text-based analysis and conclusions

## Usage

The code has been updated to automatically reference files in their organized locations:

- **API**: `SMSClassifierAPI()` automatically loads models from `models/` directory
- **Training**: `train_unified.py` saves models to `models/` and visualizations to `visualizations/`
- **Metrics**: `generate_metrics_report.py` reads models from `models/` directory

## Benefits of Organization

1. **Clear separation**: Easy to find specific types of files
2. **Scalability**: Easy to add new models or visualizations
3. **Maintenance**: Simpler to manage and backup different file types
4. **Clarity**: Clear understanding of what each directory contains
5. **Automation**: Code automatically creates subdirectories as needed

## Model Performance Summary

Based on the detailed metrics report:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Naive Bayes | 98.69% | 96.36% | 93.81% | 95.07% | 99.11% |
| Logistic Regression | 97.37% | 94.17% | 85.84% | 89.81% | 99.18% |
| Random Forest | 97.73% | 100.00% | 83.19% | 90.82% | 99.79% |

**Best performing model**: Naive Bayes (highest accuracy and F1-score)
