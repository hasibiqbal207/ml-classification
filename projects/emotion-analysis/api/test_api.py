#!/usr/bin/env python3
"""
Test script for GoEmotions Multilabel Classification API

This script runs comprehensive tests on the API endpoints.
"""

import requests
import json
import time
import sys

# API base URL
API_BASE_URL = "http://localhost:8000"

def test_endpoint(method, endpoint, data=None, expected_status=200):
    """Test a single API endpoint."""
    try:
        if method.upper() == "GET":
            response = requests.get(f"{API_BASE_URL}{endpoint}")
        elif method.upper() == "POST":
            response = requests.post(f"{API_BASE_URL}{endpoint}", json=data)
        elif method.upper() == "DELETE":
            response = requests.delete(f"{API_BASE_URL}{endpoint}")
        else:
            return False, f"Unsupported method: {method}"
        
        if response.status_code == expected_status:
            return True, response.json() if response.content else "Success"
        else:
            return False, f"Expected {expected_status}, got {response.status_code}: {response.text}"
            
    except Exception as e:
        return False, str(e)

def run_tests():
    """Run comprehensive API tests."""
    print("Running comprehensive API tests...")
    
    tests = [
        # Basic endpoints
        ("GET", "/", None, 200, "Root endpoint"),
        ("GET", "/health", None, 200, "Health check"),
        ("GET", "/models", None, 200, "List models"),
        
        # Prediction tests
        ("POST", "/predict", {"text": "I'm so happy today!", "threshold": 0.5}, 200, "Single prediction"),
        ("POST", "/predict/batch", {"texts": ["Text 1", "Text 2"], "threshold": 0.5}, 200, "Batch prediction"),
        
        # Model management tests
        ("POST", "/models/naive_bayes/load", None, 200, "Load model"),
        ("DELETE", "/models/naive_bayes/unload", None, 200, "Unload model"),
    ]
    
    passed = 0
    failed = 0
    
    for method, endpoint, data, expected_status, description in tests:
        print(f"\nTesting: {description}")
        print(f"  {method} {endpoint}")
        
        success, result = test_endpoint(method, endpoint, data, expected_status)
        
        if success:
            print(f"  ✅ PASSED")
            passed += 1
        else:
            print(f"  ❌ FAILED: {result}")
            failed += 1
    
    print(f"\n{'='*50}")
    print(f"TEST SUMMARY")
    print(f"{'='*50}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total: {passed + failed}")
    
    if failed == 0:
        print("🎉 All tests passed!")
        return True
    else:
        print("❌ Some tests failed!")
        return False

def test_error_cases():
    """Test error handling."""
    print("\nTesting error cases...")
    
    error_tests = [
        ("POST", "/predict", {"text": ""}, 422, "Empty text"),
        ("POST", "/predict", {"text": "x" * 10001}, 422, "Text too long"),
        ("POST", "/predict", {"model": "nonexistent"}, 400, "Invalid model"),
        ("POST", "/predict", {"threshold": 1.5}, 422, "Invalid threshold"),
        ("POST", "/predict/batch", {"texts": []}, 400, "Empty batch"),
        ("POST", "/predict/batch", {"texts": ["x"] * 101}, 400, "Batch too large"),
    ]
    
    for method, endpoint, data, expected_status, description in error_tests:
        print(f"\nTesting error case: {description}")
        success, result = test_endpoint(method, endpoint, data, expected_status)
        
        if success:
            print(f"  ✅ Correctly handled error")
        else:
            print(f"  ❌ Error handling failed: {result}")

def test_emotion_analysis():
    """Test emotion analysis functionality."""
    print("\nTesting emotion analysis functionality...")
    
    # Test different types of emotional content
    test_cases = [
        {
            "text": "I'm so excited and happy about this!",
            "expected_emotions": ["joy", "excitement"],
            "description": "Positive emotions"
        },
        {
            "text": "This is really frustrating and annoying.",
            "expected_emotions": ["anger", "annoyance"],
            "description": "Negative emotions"
        },
        {
            "text": "Thank you so much for your help!",
            "expected_emotions": ["gratitude"],
            "description": "Gratitude"
        },
        {
            "text": "I'm confused about what to do.",
            "expected_emotions": ["confusion"],
            "description": "Confusion"
        }
    ]
    
    for test_case in test_cases:
        try:
            response = requests.post(
                f"{API_BASE_URL}/predict",
                json={"text": test_case["text"], "threshold": 0.3}
            )
            
            if response.status_code == 200:
                result = response.json()
                predicted_emotions = [e['emotion'] for e in result['predicted_emotions'] if e['predicted']]
                print(f"  {test_case['description']}: {predicted_emotions}")
            else:
                print(f"  {test_case['description']}: Failed ({response.status_code})")
        except Exception as e:
            print(f"  {test_case['description']}: Error - {e}")

def main():
    """Main test function."""
    print("="*60)
    print("GOEMOTIONS MULTILABEL CLASSIFICATION API TESTS")
    print("="*60)
    
    # Check if API is running
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            print("❌ API is not running or not responding")
            print("Please start the API server first:")
            print("  cd api && ./start_api.sh")
            sys.exit(1)
    except requests.exceptions.RequestException:
        print("❌ Cannot connect to API")
        print("Please start the API server first:")
        print("  cd api && ./start_api.sh")
        sys.exit(1)
    
    print("✅ API is running")
    
    # Run tests
    success = run_tests()
    
    # Test error cases
    test_error_cases()
    
    # Test emotion analysis
    test_emotion_analysis()
    
    print(f"\n{'='*60}")
    if success:
        print("🎉 All tests completed successfully!")
        sys.exit(0)
    else:
        print("❌ Some tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
