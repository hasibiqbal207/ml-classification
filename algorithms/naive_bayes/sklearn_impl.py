"""
Naive Bayes Algorithm Implementation

Naive Bayes is a probabilistic classification algorithm based on Bayes' theorem
with the "naive" assumption of conditional independence between features.
"""

import numpy as np
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class NaiveBayesSklearn:
    """
    Naive Bayes implementation using Scikit-learn with multiple variants.
    """
    
    def __init__(self, variant='gaussian', alpha=1.0, fit_prior=True, 
                 class_prior=None, binarize=0.0, random_state=42):
        """
        Initialize Naive Bayes model.
        
        Args:
            variant (str): Type of Naive Bayes ('gaussian', 'multinomial', 'bernoulli')
            alpha (float): Smoothing parameter
            fit_prior (bool): Whether to learn class prior probabilities
            class_prior (array): Prior probabilities of classes
            binarize (float): Threshold for binarizing features
            random_state (int): Random state for reproducibility
        """
        self.variant = variant
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior
        self.binarize = binarize
        self.random_state = random_state
        
        # Initialize appropriate model
        if variant == 'gaussian':
            self.model = GaussianNB()
            self.scaler = StandardScaler()
        elif variant == 'multinomial':
            self.model = MultinomialNB(alpha=self.alpha, fit_prior=self.fit_prior, 
                                      class_prior=self.class_prior)
            self.scaler = MinMaxScaler()  # MultinomialNB requires non-negative features
        elif variant == 'bernoulli':
            self.model = BernoulliNB(alpha=self.alpha, fit_prior=self.fit_prior, 
                                   class_prior=self.class_prior, binarize=self.binarize)
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("Variant must be 'gaussian', 'multinomial', or 'bernoulli'")
            
        self.is_fitted = False
        
    def fit(self, X, y):
        """
        Train the Naive Bayes model.
        
        Args:
            X (array-like): Training features
            y (array-like): Training labels
            
        Returns:
            self: Fitted model
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        return self
        
    def predict(self, X):
        """
        Make predictions.
        
        Args:
            X (array-like): Features to predict
            
        Returns:
            array: Predicted labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
        
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Args:
            X (array-like): Features to predict
            
        Returns:
            array: Class probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
        
    def predict_log_proba(self, X):
        """
        Predict class log-probabilities.
        
        Args:
            X (array-like): Features to predict
            
        Returns:
            array: Class log-probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        X_scaled = self.scaler.transform(X)
        return self.model.predict_log_proba(X_scaled)
        
    def score(self, X, y):
        """
        Calculate accuracy score.
        
        Args:
            X (array-like): Features
            y (array-like): True labels
            
        Returns:
            float: Accuracy score
        """
        predictions = self.predict(X)
        return accuracy_score(y, predictions)
        
    def get_class_prior(self):
        """
        Get class prior probabilities.
        
        Returns:
            array: Class prior probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before accessing class priors")
            
        return self.model.class_prior_
        
    def get_class_count(self):
        """
        Get number of training samples observed in each class.
        
        Returns:
            array: Number of training samples per class
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before accessing class counts")
            
        return self.model.class_count_
        
    def get_feature_log_prob(self):
        """
        Get empirical log probability of features given a class.
        
        Returns:
            array: Log probability of features given each class
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before accessing feature log probabilities")
            
        return self.model.feature_log_prob_
        
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model.
        
        Args:
            X_test (array-like): Test features
            y_test (array-like): Test labels
            
        Returns:
            dict: Evaluation metrics
        """
        predictions = self.predict(X_test)
        probabilities = self.predict_proba(X_test)
        log_probabilities = self.predict_log_proba(X_test)
        
        accuracy = accuracy_score(y_test, predictions)
        
        # Get classification report
        report = classification_report(y_test, predictions, output_dict=True)
        
        # Get confusion matrix
        cm = confusion_matrix(y_test, predictions)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': predictions,
            'probabilities': probabilities,
            'log_probabilities': log_probabilities
        }
        
    def plot_class_priors(self, title="Naive Bayes Class Priors"):
        """
        Plot class prior probabilities.
        
        Args:
            title (str): Plot title
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before plotting class priors")
            
        priors = self.get_class_prior()
        classes = self.model.classes_
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(range(len(classes)), priors)
        plt.xlabel('Class')
        plt.ylabel('Prior Probability')
        plt.title(title)
        plt.xticks(range(len(classes)), classes)
        
        # Add value labels on bars
        for bar, prior in zip(bars, priors):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{prior:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
    def plot_feature_probabilities(self, feature_names=None, title="Naive Bayes Feature Probabilities"):
        """
        Plot feature probabilities for each class.
        
        Args:
            feature_names (list): Names of features
            title (str): Plot title
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before plotting feature probabilities")
            
        if self.variant != 'gaussian':
            print("Feature probability plotting is only available for Gaussian Naive Bayes")
            return
            
        # Get feature means and variances
        means = self.model.theta_
        variances = self.model.sigma_
        classes = self.model.classes_
        
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(means.shape[1])]
            
        # Plot means
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot means
        x = np.arange(len(feature_names))
        width = 0.35
        
        for i, class_name in enumerate(classes):
            ax1.bar(x + i*width, means[i], width, label=f'Class {class_name}')
            
        ax1.set_xlabel('Features')
        ax1.set_ylabel('Mean')
        ax1.set_title('Feature Means by Class')
        ax1.set_xticks(x + width/2)
        ax1.set_xticklabels(feature_names, rotation=45)
        ax1.legend()
        
        # Plot variances
        for i, class_name in enumerate(classes):
            ax2.bar(x + i*width, variances[i], width, label=f'Class {class_name}')
            
        ax2.set_xlabel('Features')
        ax2.set_ylabel('Variance')
        ax2.set_title('Feature Variances by Class')
        ax2.set_xticks(x + width/2)
        ax2.set_xticklabels(feature_names, rotation=45)
        ax2.legend()
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
        
    def plot_probability_distribution(self, X, y, feature_idx=0, title="Naive Bayes Probability Distribution"):
        """
        Plot probability distribution for a specific feature.
        
        Args:
            X (array-like): Features
            y (array-like): Labels
            feature_idx (int): Index of feature to plot
            title (str): Plot title
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before plotting probability distribution")
            
        if self.variant != 'gaussian':
            print("Probability distribution plotting is only available for Gaussian Naive Bayes")
            return
            
        # Get feature data
        feature_data = X[:, feature_idx]
        classes = self.model.classes_
        
        # Create probability distribution plot
        plt.figure(figsize=(12, 8))
        
        # Plot histogram of actual data
        for i, class_name in enumerate(classes):
            class_data = feature_data[y == class_name]
            plt.hist(class_data, alpha=0.6, label=f'Class {class_name}', bins=20)
            
        # Plot Gaussian distributions
        x_range = np.linspace(feature_data.min(), feature_data.max(), 100)
        
        for i, class_name in enumerate(classes):
            mean = self.model.theta_[i, feature_idx]
            std = np.sqrt(self.model.sigma_[i, feature_idx])
            prior = self.model.class_prior_[i]
            
            # Calculate probability density
            pdf = prior * (1/(std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_range - mean)/std)**2)
            
            plt.plot(x_range, pdf, '--', linewidth=2, label=f'Class {class_name} (PDF)')
            
        plt.xlabel(f'Feature {feature_idx}')
        plt.ylabel('Density')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
    def get_most_likely_features(self, class_idx, top_n=10):
        """
        Get most likely features for a given class.
        
        Args:
            class_idx (int): Index of class
            top_n (int): Number of top features to return
            
        Returns:
            tuple: (feature_indices, probabilities)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting most likely features")
            
        if self.variant != 'multinomial':
            print("Most likely features analysis is only available for Multinomial Naive Bayes")
            return None, None
            
        # Get feature log probabilities for the class
        feature_log_probs = self.model.feature_log_prob_[class_idx]
        
        # Get top N features
        top_indices = np.argsort(feature_log_probs)[::-1][:top_n]
        top_probs = np.exp(feature_log_probs[top_indices])
        
        return top_indices, top_probs
