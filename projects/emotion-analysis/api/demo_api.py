#!/usr/bin/env python3
"""
Demo script for GoEmotions Multilabel Classification API

This script demonstrates how to use the API for emotion analysis.
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
    """Test single text emotion prediction."""
    print("\nTesting single text emotion prediction...")
    
    # Sample texts with different emotions
    sample_texts = [
        "I'm so excited about this new project! It's going to be amazing!",
        "This is really frustrating. Nothing seems to be working properly.",
        "Thank you so much for your help. I really appreciate it!",
        "I'm feeling a bit confused about what to do next.",
        "That's hilarious! I can't stop laughing!",
        "I'm really disappointed with how this turned out.",
        "I love spending time with my family. They mean everything to me.",
        "I'm nervous about the presentation tomorrow.",
        "This is such a relief! I was worried about the results.",
        "I'm curious about how this technology works."
    ]
    
    for i, text in enumerate(sample_texts, 1):
        print(f"\nSample {i}: {text}")
        
        try:
            response = requests.post(
                f"{API_BASE_URL}/predict",
                json={"text": text, "threshold": 0.3}
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Model: {result['model_used']}")
                print(f"   Predicted emotions ({result['predicted_count']}/{result['total_emotions']}):")
                
                # Show only predicted emotions
                for emotion in result['predicted_emotions']:
                    if emotion['predicted']:
                        print(f"     {emotion['emotion']}: {emotion['confidence']:.3f}")
            else:
                print(f"❌ Prediction failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
        
        time.sleep(0.5)  # Small delay between requests

def test_batch_prediction():
    """Test batch emotion prediction."""
    print("\nTesting batch emotion prediction...")
    
    texts = [
        "I'm so happy today!",
        "This is really annoying.",
        "Thank you for everything!"
    ]
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict/batch",
            json={"texts": texts, "threshold": 0.4}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Batch prediction successful")
            for i, prediction in enumerate(result['predictions']):
                print(f"   Text {i+1}: {prediction['text'][:30]}...")
                predicted_emotions = [e['emotion'] for e in prediction['predicted_emotions'] if e['predicted']]
                print(f"     Emotions: {', '.join(predicted_emotions) if predicted_emotions else 'None'}")
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
                categories = len(model['emotion_categories']) if model['emotion_categories'] else 0
                print(f"   {model['model_name']}: {status} ({categories} emotions)")
        else:
            print(f"❌ Failed to list models: {response.status_code}")
    except Exception as e:
        print(f"❌ Failed to list models: {e}")

def test_threshold_analysis():
    """Test different threshold values."""
    print("\nTesting different threshold values...")
    
    text = "I'm really excited and happy about this news!"
    
    thresholds = [0.1, 0.3, 0.5, 0.7]
    
    for threshold in thresholds:
        try:
            response = requests.post(
                f"{API_BASE_URL}/predict",
                json={"text": text, "threshold": threshold}
            )
            
            if response.status_code == 200:
                result = response.json()
                predicted_emotions = [e['emotion'] for e in result['predicted_emotions'] if e['predicted']]
                print(f"   Threshold {threshold}: {len(predicted_emotions)} emotions - {', '.join(predicted_emotions)}")
            else:
                print(f"   Threshold {threshold}: Failed")
        except Exception as e:
            print(f"   Threshold {threshold}: Error - {e}")

def main():
    """Run all demo tests."""
    print("="*60)
    print("GOEMOTIONS MULTILABEL CLASSIFICATION API DEMO")
    print("="*60)
    
    # Test API health
    test_health()
    
    # Test single prediction
    test_prediction()
    
    # Test batch prediction
    test_batch_prediction()
    
    # Test model management
    test_model_management()
    
    # Test threshold analysis
    test_threshold_analysis()
    
    print("\n" + "="*60)
    print("Demo completed!")
    print("="*60)

if __name__ == "__main__":
    main()
