#!/usr/bin/env python3
"""
SMS Spam Detection API

A FastAPI-based real-time SMS spam detection service that loads trained models
and provides REST API endpoints for classification.
"""

import os
import sys
import pickle
import logging
import asyncio
from pathlib import Path
from typing import List, Dict, Optional, Union
from datetime import datetime
import uvicorn

# FastAPI imports
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

# ML imports
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

# Import all available algorithm classes
from algorithms.naive_bayes.sklearn_impl import NaiveBayesSklearn
from algorithms.logistic_regression.sklearn_impl import LogisticRegressionSklearn
from algorithms.random_forest.sklearn_impl import RandomForestSklearn
from algorithms.svm.sklearn_impl import SVMSklearn
from algorithms.knn.sklearn_impl import KNNSklearn
from algorithms.adaboost.sklearn_impl import AdaBoostSklearn
from algorithms.decision_tree.sklearn_impl import DecisionTreeSklearn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sms_api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Pydantic models for API
class SMSMessage(BaseModel):
    """SMS message model for API input."""
    text: str = Field(..., min_length=1, max_length=1000, description="SMS message text")
    sender_id: Optional[str] = Field(None, description="Optional sender identifier")
    timestamp: Optional[datetime] = Field(None, description="Optional message timestamp")
    
    @validator('text')
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError('Text cannot be empty or whitespace only')
        return v.strip()

class SMSBatch(BaseModel):
    """Batch of SMS messages for bulk processing."""
    messages: List[SMSMessage] = Field(..., min_items=1, max_items=100, description="List of SMS messages")
    
class PredictionResponse(BaseModel):
    """Response model for single prediction."""
    text: str
    prediction: str = Field(..., description="Predicted class: 'ham' or 'spam'")
    probability: float = Field(..., ge=0, le=1, description="Confidence score")
    model_used: str = Field(..., description="Model that made the prediction")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    sender_id: Optional[str] = None
    timestamp: Optional[datetime] = None

class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    predictions: List[PredictionResponse]
    total_messages: int
    total_processing_time_ms: float
    spam_count: int
    ham_count: int

class ModelInfo(BaseModel):
    """Model information response."""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    is_loaded: bool
    last_updated: datetime

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: datetime
    models_loaded: int
    total_models: int
    uptime_seconds: float

class SMSClassifierAPI:
    """SMS Classification API service using algorithm classes."""
    
    def __init__(self, results_dir: str = None):
        """
        Initialize the SMS Classification API.
        
        Args:
            results_dir: Path to results directory containing trained models
        """
        if results_dir is None:
            self.results_dir = Path(__file__).parent.parent / "results"
        else:
            self.results_dir = Path(results_dir)
            
        # Create results directory and subdirectories if they don't exist
        self.results_dir.mkdir(exist_ok=True)
        (self.results_dir / "models").mkdir(exist_ok=True)
        (self.results_dir / "visualizations").mkdir(exist_ok=True)
        (self.results_dir / "reports").mkdir(exist_ok=True)
            
        self.models = {}
        self.vectorizers = {}
        self.model_info = {}
        self.start_time = datetime.now()
        
        # Algorithm class mapping
        self.algorithm_classes = {
            'naive_bayes': NaiveBayesSklearn,
            'logistic_regression': LogisticRegressionSklearn,
            'random_forest': RandomForestSklearn,
            'svm': SVMSklearn,
            'knn': KNNSklearn,
            'adaboost': AdaBoostSklearn,
            'decision_tree': DecisionTreeSklearn
        }
        
        # Load all available models
        self.load_models()
        
    def load_models(self):
        """Load all available trained models using algorithm classes."""
        logger.info("Loading trained models using algorithm classes...")
        
        # Check for all available algorithm models
        for model_name, algorithm_class in self.algorithm_classes.items():
            try:
                model_path = self.results_dir / "models" / f"{model_name}_model.pkl"
                vectorizer_path = self.results_dir / "models" / f"{model_name}_vectorizer.pkl"
                
                if model_path.exists() and vectorizer_path.exists():
                    # Create algorithm instance
                    algorithm_instance = algorithm_class()
                    
                    # Load model using algorithm class's built-in method
                    if hasattr(algorithm_instance, 'load_model'):
                        algorithm_instance.load_model(str(model_path))
                    else:
                        # Fallback to manual loading
                        with open(model_path, 'rb') as f:
                            algorithm_instance = pickle.load(f)
                    
                    # Load vectorizer
                    with open(vectorizer_path, 'rb') as f:
                        vectorizer = pickle.load(f)
                    
                    # Store model and vectorizer
                    self.models[model_name] = algorithm_instance
                    self.vectorizers[model_name] = vectorizer
                    
                    # Load model performance info
                    self.model_info[model_name] = self._get_model_performance(model_name)
                    
                    logger.info(f"Successfully loaded {model_name} model using {algorithm_class.__name__}")
                else:
                    logger.warning(f"Model files not found for {model_name}")
                    
            except Exception as e:
                logger.error(f"Error loading {model_name} model: {e}")
                
        logger.info(f"Loaded {len(self.models)} models successfully")
        
    def _get_model_performance(self, model_name: str) -> Dict:
        """Get model performance metrics."""
        # These would ideally be loaded from a metrics file or computed dynamically
        # For now, using known performance from training
        performance_data = {
            'naive_bayes': {
                'accuracy': 0.9869,
                'precision': 0.9636,
                'recall': 0.9381,
                'f1_score': 0.9507,
                'roc_auc': 0.9911
            },
            'logistic_regression': {
                'accuracy': 0.9737,
                'precision': 0.9417,
                'recall': 0.8584,
                'f1_score': 0.8981,
                'roc_auc': 0.9918
            },
            'random_forest': {
                'accuracy': 0.9773,
                'precision': 1.0000,
                'recall': 0.8319,
                'f1_score': 0.9082,
                'roc_auc': 0.9979
            },
            'svm': {
                'accuracy': 0.0,  # Will be updated after training
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'roc_auc': 0.0
            },
            'knn': {
                'accuracy': 0.0,  # Will be updated after training
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'roc_auc': 0.0
            },
            'adaboost': {
                'accuracy': 0.0,  # Will be updated after training
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'roc_auc': 0.0
            },
            'decision_tree': {
                'accuracy': 0.0,  # Will be updated after training
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'roc_auc': 0.0
            }
        }
        
        return performance_data.get(model_name, {})
        
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess SMS text (same as training preprocessing).
        
        Args:
            text: Raw SMS text
            
        Returns:
            Preprocessed text
        """
        import re
        
        # Convert to lowercase
        text = text.lower()
        
        # Replace URLs with placeholder
        url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        text = url_pattern.sub(' <url> ', text)
        
        # Replace email addresses with placeholder
        email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        text = email_pattern.sub(' <email> ', text)
        
        # Replace phone numbers with placeholder
        phone_pattern = re.compile(r'\b\d{10,}\b')
        text = phone_pattern.sub(' <phone> ', text)
        
        # Replace special characters and normalize whitespace
        special_chars_pattern = re.compile(r'[^a-zA-Z0-9\s]')
        text = special_chars_pattern.sub(' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
        
    def predict_single_sync(self, message: SMSMessage, model_name: str = 'naive_bayes') -> PredictionResponse:
        """
        Synchronous version of predict_single using algorithm classes.
        
        Args:
            message: SMS message to classify
            model_name: Model to use for prediction
            
        Returns:
            Prediction response
        """
        start_time = datetime.now()
        
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not available")
            
        try:
            # Preprocess text
            processed_text = self.preprocess_text(message.text)
            
            # Transform text using vectorizer
            vectorizer = self.vectorizers[model_name]
            X = vectorizer.transform([processed_text])
            
            # Make prediction using algorithm class
            model = self.models[model_name]
            
            # Use algorithm class's predict method
            prediction = model.predict(X.toarray())[0]
            
            # Use algorithm class's predict_proba method
            if hasattr(model, 'predict_proba'):
                probability = model.predict_proba(X.toarray())[0]
            else:
                # Fallback for models without probability prediction
                probability = [0.5, 0.5]  # Default probabilities
            
            # Convert prediction to string
            prediction_str = 'spam' if prediction == 1 else 'ham'
            confidence = probability[1] if prediction == 1 else probability[0]
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return PredictionResponse(
                text=message.text,
                prediction=prediction_str,
                probability=confidence,
                model_used=model_name,
                processing_time_ms=processing_time,
                sender_id=message.sender_id,
                timestamp=message.timestamp or datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            raise ValueError(f"Prediction failed: {str(e)}")
            
    async def predict_single(self, message: SMSMessage, model_name: str = 'naive_bayes') -> PredictionResponse:
        """
        Predict spam/ham for a single SMS message.
        
        Args:
            message: SMS message to classify
            model_name: Model to use for prediction
            
        Returns:
            Prediction response
        """
        start_time = datetime.now()
        
        if model_name not in self.models:
            raise HTTPException(status_code=400, detail=f"Model '{model_name}' not available")
            
        try:
            # Preprocess text
            processed_text = self.preprocess_text(message.text)
            
            # Transform text using vectorizer
            vectorizer = self.vectorizers[model_name]
            X = vectorizer.transform([processed_text])
            
            # Make prediction
            model = self.models[model_name]
            prediction = model.predict(X.toarray())[0]
            probability = model.predict_proba(X.toarray())[0]
            
            # Convert prediction to string
            prediction_str = 'spam' if prediction == 1 else 'ham'
            confidence = probability[1] if prediction == 1 else probability[0]
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return PredictionResponse(
                text=message.text,
                prediction=prediction_str,
                probability=confidence,
                model_used=model_name,
                processing_time_ms=processing_time,
                sender_id=message.sender_id,
                timestamp=message.timestamp or datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
            
    async def predict_batch(self, batch: SMSBatch, model_name: str = 'naive_bayes') -> BatchPredictionResponse:
        """
        Predict spam/ham for a batch of SMS messages.
        
        Args:
            batch: Batch of SMS messages
            model_name: Model to use for prediction
            
        Returns:
            Batch prediction response
        """
        start_time = datetime.now()
        
        if model_name not in self.models:
            raise HTTPException(status_code=400, detail=f"Model '{model_name}' not available")
            
        try:
            predictions = []
            
            for message in batch.messages:
                prediction = await self.predict_single(message, model_name)
                predictions.append(prediction)
            
            # Calculate batch statistics
            total_time = (datetime.now() - start_time).total_seconds() * 1000
            spam_count = sum(1 for p in predictions if p.prediction == 'spam')
            ham_count = len(predictions) - spam_count
            
            return BatchPredictionResponse(
                predictions=predictions,
                total_messages=len(predictions),
                total_processing_time_ms=total_time,
                spam_count=spam_count,
                ham_count=ham_count
            )
            
        except Exception as e:
            logger.error(f"Error in batch prediction: {e}")
            raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

# Initialize API service
api_service = SMSClassifierAPI()

# Create FastAPI app
app = FastAPI(
    title="SMS Spam Detection API",
    description="Real-time SMS spam detection using machine learning models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "SMS Spam Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    uptime = (datetime.now() - api_service.start_time).total_seconds()
    
    return HealthResponse(
        status="healthy" if len(api_service.models) > 0 else "degraded",
        timestamp=datetime.now(),
        models_loaded=len(api_service.models),
        total_models=len(api_service.algorithm_classes),
        uptime_seconds=uptime
    )

@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """List all available models and their performance."""
    models_info = []
    
    for model_name, info in api_service.model_info.items():
        models_info.append(ModelInfo(
            model_name=model_name,
            accuracy=info.get('accuracy', 0.0),
            precision=info.get('precision', 0.0),
            recall=info.get('recall', 0.0),
            f1_score=info.get('f1_score', 0.0),
            roc_auc=info.get('roc_auc', 0.0),
            is_loaded=model_name in api_service.models,
            last_updated=datetime.now()
        ))
    
    return models_info

@app.post("/predict", response_model=PredictionResponse)
async def predict_sms(message: SMSMessage, model: str = 'naive_bayes'):
    """
    Predict if a single SMS message is spam or ham.
    
    - **text**: SMS message text (required)
    - **sender_id**: Optional sender identifier
    - **timestamp**: Optional message timestamp
    - **model**: Model to use ('naive_bayes', 'logistic_regression', 'random_forest')
    """
    return await api_service.predict_single(message, model)

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_sms_batch(batch: SMSBatch, model: str = 'naive_bayes'):
    """
    Predict spam/ham for multiple SMS messages in a single request.
    
    - **messages**: List of SMS messages (max 100)
    - **model**: Model to use ('naive_bayes', 'logistic_regression', 'random_forest')
    """
    return await api_service.predict_batch(batch, model)

@app.post("/predict/ensemble", response_model=PredictionResponse)
async def predict_ensemble(message: SMSMessage):
    """
    Predict using ensemble of all available models.
    
    - **text**: SMS message text (required)
    - **sender_id**: Optional sender identifier
    - **timestamp**: Optional message timestamp
    """
    if not api_service.models:
        raise HTTPException(status_code=500, detail="No models available")
    
    start_time = datetime.now()
    
    try:
        # Get predictions from all models
        predictions = []
        probabilities = []
        
        for model_name in api_service.models.keys():
            pred = await api_service.predict_single(message, model_name)
            predictions.append(1 if pred.prediction == 'spam' else 0)
            probabilities.append(pred.probability)
        
        # Ensemble prediction (majority vote)
        spam_votes = sum(predictions)
        total_votes = len(predictions)
        
        # Final prediction
        final_prediction = 'spam' if spam_votes > total_votes / 2 else 'ham'
        
        # Average probability
        avg_probability = np.mean(probabilities)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return PredictionResponse(
            text=message.text,
            prediction=final_prediction,
            probability=avg_probability,
            model_used=f"ensemble_{total_votes}_models",
            processing_time_ms=processing_time,
            sender_id=message.sender_id,
            timestamp=message.timestamp or datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error in ensemble prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Ensemble prediction failed: {str(e)}")

@app.get("/stats")
async def get_stats():
    """Get API usage statistics."""
    return {
        "models_loaded": len(api_service.models),
        "total_available": len(api_service.algorithm_classes),
        "available_models": list(api_service.models.keys()),
        "all_algorithm_classes": list(api_service.algorithm_classes.keys()),
        "uptime_seconds": (datetime.now() - api_service.start_time).total_seconds(),
        "api_version": "1.0.0"
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "timestamp": datetime.now().isoformat()}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "timestamp": datetime.now().isoformat()}
    )

if __name__ == "__main__":
    # Run the API server
    uvicorn.run(
        "sms_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
