#!/usr/bin/env python3
"""
FastAPI Application for 20 Newsgroups Classification

This API provides real-time newsgroup classification using trained models.
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
    title="20 Newsgroups Classification API",
    description="Real-time newsgroup classification using machine learning models",
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
    text: str = Field(..., description="Text to classify", min_length=1, max_length=10000)
    model: Optional[str] = Field(None, description="Specific model to use")

class ClassificationResponse(BaseModel):
    predicted_category: str
    confidence: float
    model_used: str
    all_predictions: Dict[str, float]

class HealthResponse(BaseModel):
    status: str
    available_models: List[str]
    loaded_models: List[str]

class ModelInfo(BaseModel):
    model_name: str
    is_loaded: bool
    categories: Optional[List[str]] = None

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
        encoder_path = results_dir / "models" / f"{model_name}_label_encoder.pkl"
        
        if not all([model_path.exists(), vectorizer_path.exists(), encoder_path.exists()]):
            raise FileNotFoundError(f"Model files not found for {model_name}")
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        
        with open(encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
        
        return {
            'model': model,
            'vectorizer': vectorizer,
            'label_encoder': label_encoder
        }
        
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load model {model_name}: {str(e)}")

def preprocess_text(text: str) -> str:
    """
    Preprocess text for classification.
    
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
    
    # Replace special characters and normalize whitespace
    special_chars_pattern = re.compile(r'[^a-zA-Z0-9\s]')
    text = special_chars_pattern.sub(' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def predict_category(text: str, model_name: str) -> Dict[str, Any]:
    """
    Predict category for given text using specified model.
    
    Args:
        text: Text to classify
        model_name: Name of the model to use
        
    Returns:
        dict: Prediction results
    """
    if model_name not in loaded_models:
        # Load model if not already loaded
        loaded_models[model_name] = load_model(model_name)
    
    model_components = loaded_models[model_name]
    model = model_components['model']
    vectorizer = model_components['vectorizer']
    label_encoder = model_components['label_encoder']
    
    # Preprocess text
    processed_text = preprocess_text(text)
    
    # Transform text
    X = vectorizer.transform([processed_text])
    
    # Make prediction
    prediction = model.predict(X.toarray())[0]
    predicted_category = label_encoder.inverse_transform([prediction])[0]
    
    # Get prediction probabilities if available
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X.toarray())[0]
        confidence = probabilities[prediction]
        
        # Get all category probabilities
        all_predictions = {}
        for i, category in enumerate(label_encoder.classes_):
            all_predictions[category] = float(probabilities[i])
    else:
        confidence = 1.0  # Fallback for models without probability
        all_predictions = {predicted_category: 1.0}
    
    return {
        'predicted_category': predicted_category,
        'confidence': float(confidence),
        'model_used': model_name,
        'all_predictions': all_predictions
    }

# API Endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "20 Newsgroups Classification API",
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
        categories = None
        
        if is_loaded:
            try:
                label_encoder = loaded_models[model_name]['label_encoder']
                categories = label_encoder.classes_.tolist()
            except:
                pass
        
        models_info.append(ModelInfo(
            model_name=model_name,
            is_loaded=is_loaded,
            categories=categories
        ))
    
    return models_info

@app.post("/predict", response_model=ClassificationResponse)
async def predict(input_data: TextInput):
    """
    Predict newsgroup category for given text.
    
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
        result = predict_category(input_data.text, model_name)
        
        return ClassificationResponse(**result)
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
async def predict_batch(texts: List[str], model: Optional[str] = None):
    """
    Predict categories for multiple texts.
    
    Args:
        texts: List of texts to classify
        model: Optional model name
        
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
            result = predict_category(text, model_name)
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
    logger.info("Starting 20 Newsgroups Classification API...")
    
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
        "newsgroups_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
