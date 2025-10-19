#!/usr/bin/env python3
"""
FastAPI Application for GoEmotions Multilabel Classification

This API provides real-time emotion analysis using trained models.
"""

import os
import sys
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="GoEmotions Multilabel Classification API",
    description="Real-time emotion analysis using machine learning models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for loaded models
loaded_models = {}
available_models = [
    'naive_bayes', 'logistic_regression', 'random_forest', 
    'svm', 'knn', 'adaboost', 'decision_tree'
]

# Pydantic models
class TextInput(BaseModel):
    text: str = Field(..., description="Text to analyze for emotions", min_length=1, max_length=10000)
    model: Optional[str] = Field(None, description="Specific model to use")
    threshold: Optional[float] = Field(0.5, description="Confidence threshold for emotion predictions", ge=0.0, le=1.0)

class EmotionPrediction(BaseModel):
    emotion: str
    confidence: float
    predicted: bool

class ClassificationResponse(BaseModel):
    text: str
    predicted_emotions: List[EmotionPrediction]
    model_used: str
    threshold: float
    total_emotions: int
    predicted_count: int

class HealthResponse(BaseModel):
    status: str
    available_models: List[str]
    loaded_models: List[str]

class ModelInfo(BaseModel):
    model_name: str
    is_loaded: bool
    emotion_categories: Optional[List[str]] = None

# Utility functions
def load_model(model_name: str) -> Dict[str, Any]:
    """
    Load a specific model and its components.
    
    Args:
        model_name: Name of the model to load
        
    Returns:
        dict: Model components
    """
    try:
        results_dir = Path(__file__).parent.parent / "results"
        
        model_path = results_dir / "models" / f"{model_name}_model.pkl"
        vectorizer_path = results_dir / "models" / f"{model_name}_vectorizer.pkl"
        categories_path = results_dir / "models" / f"{model_name}_emotion_categories.pkl"
        
        if not all([model_path.exists(), vectorizer_path.exists(), categories_path.exists()]):
            raise FileNotFoundError(f"Model files not found for {model_name}")
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        
        with open(categories_path, 'rb') as f:
            emotion_categories = pickle.load(f)
        
        return {
            'model': model,
            'vectorizer': vectorizer,
            'emotion_categories': emotion_categories
        }
        
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load model {model_name}: {str(e)}")

def preprocess_text(text: str) -> str:
    """
    Preprocess text for emotion analysis.
    
    Args:
        text: Raw text
        
    Returns:
        str: Preprocessed text
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
    
    # Remove Reddit-specific patterns
    text = re.sub(r'\[.*?\]', '', text)  # Remove [NAME], [RELIGION], etc.
    text = re.sub(r'u/[a-zA-Z0-9_]+', '<user>', text)  # Replace usernames
    text = re.sub(r'r/[a-zA-Z0-9_]+', '<subreddit>', text)  # Replace subreddits
    
    # Replace special characters and normalize whitespace
    special_chars_pattern = re.compile(r'[^a-zA-Z0-9\s]')
    text = special_chars_pattern.sub(' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def predict_emotions(text: str, model_name: str, threshold: float = 0.5) -> Dict[str, Any]:
    """
    Predict emotions for given text using specified model.
    
    Args:
        text: Text to analyze
        model_name: Name of the model to use
        threshold: Confidence threshold for predictions
        
    Returns:
        dict: Prediction results
    """
    if model_name not in loaded_models:
        # Load model if not already loaded
        loaded_models[model_name] = load_model(model_name)
    
    model_components = loaded_models[model_name]
    model = model_components['model']
    vectorizer = model_components['vectorizer']
    emotion_categories = model_components['emotion_categories']
    
    # Preprocess text
    processed_text = preprocess_text(text)
    
    # Transform text
    X = vectorizer.transform([processed_text])
    
    # Make prediction probabilities
    probabilities = model.predict_proba(X)[0]
    
    # Create emotion predictions
    predicted_emotions = []
    predicted_count = 0
    
    for i, emotion in enumerate(emotion_categories):
        confidence = float(probabilities[i])
        predicted = confidence >= threshold
        
        predicted_emotions.append(EmotionPrediction(
            emotion=emotion,
            confidence=confidence,
            predicted=predicted
        ))
        
        if predicted:
            predicted_count += 1
    
    return {
        'text': text,
        'predicted_emotions': predicted_emotions,
        'model_used': model_name,
        'threshold': threshold,
        'total_emotions': len(emotion_categories),
        'predicted_count': predicted_count
    }

# API Endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "GoEmotions Multilabel Classification API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        available_models=available_models,
        loaded_models=list(loaded_models.keys())
    )

@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """List all available models and their status."""
    models_info = []
    
    for model_name in available_models:
        is_loaded = model_name in loaded_models
        emotion_categories = None
        
        if is_loaded:
            try:
                emotion_categories = loaded_models[model_name]['emotion_categories']
            except:
                pass
        
        models_info.append(ModelInfo(
            model_name=model_name,
            is_loaded=is_loaded,
            emotion_categories=emotion_categories
        ))
    
    return models_info

@app.post("/predict", response_model=ClassificationResponse)
async def predict(input_data: TextInput):
    """
    Predict emotions for given text.
    
    Args:
        input_data: Text input and optional model specification
        
    Returns:
        ClassificationResponse: Prediction results
    """
    try:
        # Use specified model or default to first available loaded model
        if input_data.model:
            if input_data.model not in available_models:
                raise HTTPException(status_code=400, detail=f"Model '{input_data.model}' not available")
            model_name = input_data.model
        else:
            if not loaded_models:
                raise HTTPException(status_code=400, detail="No models loaded. Please specify a model or load models first.")
            model_name = list(loaded_models.keys())[0]
        
        # Make prediction
        result = predict_emotions(input_data.text, model_name, input_data.threshold)
        
        return ClassificationResponse(**result)
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
async def predict_batch(texts: List[str], model: Optional[str] = None, threshold: float = 0.5):
    """
    Predict emotions for multiple texts.
    
    Args:
        texts: List of texts to analyze
        model: Optional model name
        threshold: Confidence threshold for predictions
        
    Returns:
        List of prediction results
    """
    try:
        if not texts:
            raise HTTPException(status_code=400, detail="No texts provided")
        
        if len(texts) > 100:
            raise HTTPException(status_code=400, detail="Too many texts. Maximum 100 texts per batch.")
        
        # Use specified model or default
        if model:
            if model not in available_models:
                raise HTTPException(status_code=400, detail=f"Model '{model}' not available")
            model_name = model
        else:
            if not loaded_models:
                raise HTTPException(status_code=400, detail="No models loaded")
            model_name = list(loaded_models.keys())[0]
        
        results = []
        for text in texts:
            result = predict_emotions(text, model_name, threshold)
            results.append(result)
        
        return {"predictions": results}
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/models/{model_name}/load")
async def load_model_endpoint(model_name: str):
    """
    Load a specific model.
    
    Args:
        model_name: Name of the model to load
        
    Returns:
        dict: Load status
    """
    try:
        if model_name not in available_models:
            raise HTTPException(status_code=400, detail=f"Model '{model_name}' not available")
        
        loaded_models[model_name] = load_model(model_name)
        
        return {
            "message": f"Model '{model_name}' loaded successfully",
            "loaded_models": list(loaded_models.keys())
        }
        
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/models/{model_name}/unload")
async def unload_model_endpoint(model_name: str):
    """
    Unload a specific model.
    
    Args:
        model_name: Name of the model to unload
        
    Returns:
        dict: Unload status
    """
    try:
        if model_name in loaded_models:
            del loaded_models[model_name]
            return {
                "message": f"Model '{model_name}' unloaded successfully",
                "loaded_models": list(loaded_models.keys())
            }
        else:
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not loaded")
            
    except Exception as e:
        logger.error(f"Failed to unload model {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/models/load-all")
async def load_all_models():
    """
    Load all available models.
    
    Returns:
        dict: Load status for all models
    """
    results = {}
    
    for model_name in available_models:
        try:
            loaded_models[model_name] = load_model(model_name)
            results[model_name] = "loaded"
        except Exception as e:
            results[model_name] = f"failed: {str(e)}"
    
    return {
        "message": "Batch model loading completed",
        "results": results,
        "loaded_models": list(loaded_models.keys())
    }

# Startup event
@app.on_event("startup")
async def startup_event():
    """Load default models on startup."""
    logger.info("Starting GoEmotions Multilabel Classification API...")
    
    # Try to load the first available model
    for model_name in available_models:
        try:
            loaded_models[model_name] = load_model(model_name)
            logger.info(f"Loaded model: {model_name}")
            break  # Load only one model by default
        except Exception as e:
            logger.warning(f"Failed to load model {model_name}: {e}")
    
    if not loaded_models:
        logger.warning("No models loaded on startup")

if __name__ == "__main__":
    uvicorn.run(
        "emotion_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
