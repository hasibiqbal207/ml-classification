#!/usr/bin/env python3
"""
Demo script for 20 Newsgroups Classification API

This script demonstrates how to use the API for newsgroup classification.
"""

import requests
import json
import time

# API base URL
API_BASE_URL = "http://localhost:8000"

def test_health():
    """Test API health endpoint."""
    print("Testing API health...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API is healthy")
            print(f"Available models: {data['available_models']}")
            print(f"Loaded models: {data['loaded_models']}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")

def test_prediction():
    """Test single text prediction."""
    print("\nTesting single text prediction...")
    
    # Sample texts from different newsgroups
    sample_texts = [
        "I'm looking for information about graphics programming and OpenGL. Can anyone recommend good tutorials?",
        "My computer keeps crashing when I try to run Windows applications. Any suggestions?",
        "I'm selling my old MacBook Pro. It's in good condition and comes with all accessories.",
        "The Lakers played an amazing game last night. LeBron's performance was incredible!",
        "I'm having trouble with my car's engine. It's making strange noises when I accelerate.",
        "What are your thoughts on the recent political developments in the Middle East?",
        "I'm interested in learning about cryptography and encryption methods.",
        "The Hubble Space Telescope has captured some amazing images of distant galaxies."
    ]
    
    for i, text in enumerate(sample_texts, 1):
        print(f"\nSample {i}: {text[:50]}...")
        
        try:
            response = requests.post(
                f"{API_BASE_URL}/predict",
                json={"text": text}
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Predicted category: {result['predicted_category']}")
                print(f"   Confidence: {result['confidence']:.3f}")
                print(f"   Model used: {result['model_used']}")
            else:
                print(f"❌ Prediction failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
        
        time.sleep(0.5)  # Small delay between requests

def test_batch_prediction():
    """Test batch prediction."""
    print("\nTesting batch prediction...")
    
    texts = [
        "Looking for advice on computer graphics programming",
        "Selling my old MacBook in excellent condition",
        "What's the latest news in space exploration?"
    ]
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict/batch",
            json={"texts": texts}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Batch prediction successful")
            for i, prediction in enumerate(result['predictions']):
                print(f"   Text {i+1}: {prediction['predicted_category']} (confidence: {prediction['confidence']:.3f})")
        else:
            print(f"❌ Batch prediction failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Batch prediction failed: {e}")

def test_model_management():
    """Test model loading/unloading."""
    print("\nTesting model management...")
    
    # List models
    try:
        response = requests.get(f"{API_BASE_URL}/models")
        if response.status_code == 200:
            models = response.json()
            print(f"✅ Available models: {len(models)}")
            for model in models:
                status = "loaded" if model['is_loaded'] else "not loaded"
                print(f"   {model['model_name']}: {status}")
        else:
            print(f"❌ Failed to list models: {response.status_code}")
    except Exception as e:
        print(f"❌ Failed to list models: {e}")

def main():
    """Run all demo tests."""
    print("="*60)
    print("20 NEWSCROUPS CLASSIFICATION API DEMO")
    print("="*60)
    
    # Test API health
    test_health()
    
    # Test single prediction
    test_prediction()
    
    # Test batch prediction
    test_batch_prediction()
    
    # Test model management
    test_model_management()
    
    print("\n" + "="*60)
    print("Demo completed!")
    print("="*60)

if __name__ == "__main__":
    main()
