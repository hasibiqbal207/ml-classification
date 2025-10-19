#!/usr/bin/env python3
"""
API Testing Script for SMS Spam Detection API

This script provides comprehensive testing for the SMS classification API,
including unit tests, integration tests, and performance benchmarks.
"""

import asyncio
import time
import requests
import json
from datetime import datetime
from typing import List, Dict
import statistics

# Test data
TEST_MESSAGES = [
    {
        "text": "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question",
        "expected": "spam"
    },
    {
        "text": "Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat",
        "expected": "ham"
    },
    {
        "text": "WINNER!! As a valued network customer you have been selected to receivea £900 prize reward! To claim call 09061701461",
        "expected": "spam"
    },
    {
        "text": "Ok lar... Joking wif u oni...",
        "expected": "ham"
    },
    {
        "text": "URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot! Txt the word: CLAIM to No: 81010",
        "expected": "spam"
    },
    {
        "text": "I've been searching for the right words to thank you for this breather. I promise i wont take your help for granted",
        "expected": "ham"
    }
]

class APITester:
    """Comprehensive API testing class."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize API tester.
        
        Args:
            base_url: Base URL of the API
        """
        self.base_url = base_url
        self.session = requests.Session()
        
    def test_health_check(self) -> bool:
        """Test health check endpoint."""
        print("Testing health check endpoint...")
        try:
            response = self.session.get(f"{self.base_url}/health")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Health check passed: {data['status']}")
                print(f"   Models loaded: {data['models_loaded']}/{data['total_models']}")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False
    
    def test_models_endpoint(self) -> bool:
        """Test models listing endpoint."""
        print("\nTesting models endpoint...")
        try:
            response = self.session.get(f"{self.base_url}/models")
            if response.status_code == 200:
                models = response.json()
                print(f"✅ Models endpoint passed: {len(models)} models found")
                for model in models:
                    print(f"   - {model['model_name']}: Accuracy={model['accuracy']:.4f}, Loaded={model['is_loaded']}")
                return True
            else:
                print(f"❌ Models endpoint failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Models endpoint error: {e}")
            return False
    
    def test_single_prediction(self, model_name: str = "naive_bayes") -> bool:
        """Test single SMS prediction endpoint."""
        print(f"\nTesting single prediction with {model_name} model...")
        
        test_message = {
            "text": "Free entry in 2 a wkly comp to win FA Cup final tkts",
            "sender_id": "test_sender",
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/predict",
                params={"model": model_name},
                json=test_message
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Single prediction passed")
                print(f"   Text: {data['text'][:50]}...")
                print(f"   Prediction: {data['prediction']}")
                print(f"   Probability: {data['probability']:.4f}")
                print(f"   Model: {data['model_used']}")
                print(f"   Processing time: {data['processing_time_ms']:.2f}ms")
                return True
            else:
                print(f"❌ Single prediction failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
        except Exception as e:
            print(f"❌ Single prediction error: {e}")
            return False
    
    def test_batch_prediction(self, model_name: str = "naive_bayes") -> bool:
        """Test batch SMS prediction endpoint."""
        print(f"\nTesting batch prediction with {model_name} model...")
        
        batch_messages = {
            "messages": [
                {"text": msg["text"], "sender_id": f"sender_{i}"}
                for i, msg in enumerate(TEST_MESSAGES[:3])
            ]
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/predict/batch",
                params={"model": model_name},
                json=batch_messages
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Batch prediction passed")
                print(f"   Total messages: {data['total_messages']}")
                print(f"   Spam count: {data['spam_count']}")
                print(f"   Ham count: {data['ham_count']}")
                print(f"   Total processing time: {data['total_processing_time_ms']:.2f}ms")
                return True
            else:
                print(f"❌ Batch prediction failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
        except Exception as e:
            print(f"❌ Batch prediction error: {e}")
            return False
    
    def test_ensemble_prediction(self) -> bool:
        """Test ensemble prediction endpoint."""
        print("\nTesting ensemble prediction...")
        
        test_message = {
            "text": "Free entry in 2 a wkly comp to win FA Cup final tkts",
            "sender_id": "test_sender"
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/predict/ensemble",
                json=test_message
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Ensemble prediction passed")
                print(f"   Prediction: {data['prediction']}")
                print(f"   Probability: {data['probability']:.4f}")
                print(f"   Model: {data['model_used']}")
                print(f"   Processing time: {data['processing_time_ms']:.2f}ms")
                return True
            else:
                print(f"❌ Ensemble prediction failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
        except Exception as e:
            print(f"❌ Ensemble prediction error: {e}")
            return False
    
    def test_accuracy(self, model_name: str = "naive_bayes") -> Dict:
        """Test prediction accuracy on known test cases."""
        print(f"\nTesting accuracy with {model_name} model...")
        
        correct_predictions = 0
        total_predictions = len(TEST_MESSAGES)
        results = []
        
        for i, test_case in enumerate(TEST_MESSAGES):
            try:
                response = self.session.post(
                    f"{self.base_url}/predict",
                    params={"model": model_name},
                    json={"text": test_case["text"]}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    prediction = data["prediction"]
                    expected = test_case["expected"]
                    
                    is_correct = prediction == expected
                    if is_correct:
                        correct_predictions += 1
                    
                    results.append({
                        "text": test_case["text"][:50] + "...",
                        "expected": expected,
                        "predicted": prediction,
                        "correct": is_correct,
                        "probability": data["probability"]
                    })
                    
                    print(f"   Test {i+1}: {'✅' if is_correct else '❌'} "
                          f"Expected: {expected}, Got: {prediction}")
                else:
                    print(f"   Test {i+1}: ❌ API Error: {response.status_code}")
                    
            except Exception as e:
                print(f"   Test {i+1}: ❌ Error: {e}")
        
        accuracy = correct_predictions / total_predictions
        print(f"✅ Accuracy test completed: {correct_predictions}/{total_predictions} ({accuracy:.2%})")
        
        return {
            "accuracy": accuracy,
            "correct_predictions": correct_predictions,
            "total_predictions": total_predictions,
            "results": results
        }
    
    def benchmark_performance(self, num_requests: int = 100) -> Dict:
        """Benchmark API performance."""
        print(f"\nBenchmarking performance with {num_requests} requests...")
        
        test_message = {"text": "Free entry in 2 a wkly comp to win FA Cup final tkts"}
        response_times = []
        successful_requests = 0
        
        start_time = time.time()
        
        for i in range(num_requests):
            try:
                request_start = time.time()
                response = self.session.post(
                    f"{self.base_url}/predict",
                    params={"model": "naive_bayes"},
                    json=test_message
                )
                request_end = time.time()
                
                if response.status_code == 200:
                    successful_requests += 1
                    response_times.append((request_end - request_start) * 1000)  # Convert to ms
                
                if (i + 1) % 20 == 0:
                    print(f"   Completed {i + 1}/{num_requests} requests...")
                    
            except Exception as e:
                print(f"   Request {i + 1} failed: {e}")
        
        total_time = time.time() - start_time
        
        if response_times:
            stats = {
                "total_requests": num_requests,
                "successful_requests": successful_requests,
                "success_rate": successful_requests / num_requests,
                "total_time_seconds": total_time,
                "requests_per_second": num_requests / total_time,
                "avg_response_time_ms": statistics.mean(response_times),
                "min_response_time_ms": min(response_times),
                "max_response_time_ms": max(response_times),
                "median_response_time_ms": statistics.median(response_times)
            }
            
            print(f"✅ Performance benchmark completed:")
            print(f"   Success rate: {stats['success_rate']:.2%}")
            print(f"   Requests per second: {stats['requests_per_second']:.2f}")
            print(f"   Average response time: {stats['avg_response_time_ms']:.2f}ms")
            print(f"   Median response time: {stats['median_response_time_ms']:.2f}ms")
            
            return stats
        else:
            print("❌ No successful requests for benchmarking")
            return {}
    
    def run_all_tests(self) -> bool:
        """Run all API tests."""
        print("=" * 60)
        print("SMS SPAM DETECTION API - COMPREHENSIVE TESTING")
        print("=" * 60)
        
        tests_passed = 0
        total_tests = 6
        
        # Basic endpoint tests
        if self.test_health_check():
            tests_passed += 1
        
        if self.test_models_endpoint():
            tests_passed += 1
        
        # Prediction tests
        if self.test_single_prediction():
            tests_passed += 1
        
        if self.test_batch_prediction():
            tests_passed += 1
        
        if self.test_ensemble_prediction():
            tests_passed += 1
        
        # Accuracy test
        accuracy_results = self.test_accuracy()
        if accuracy_results["accuracy"] > 0.8:  # Expect at least 80% accuracy
            tests_passed += 1
        
        # Performance benchmark
        performance_results = self.benchmark_performance(50)  # Reduced for faster testing
        if performance_results.get("success_rate", 0) > 0.9:  # Expect at least 90% success rate
            tests_passed += 1
        
        print("\n" + "=" * 60)
        print(f"TESTING COMPLETED: {tests_passed}/{total_tests} tests passed")
        print("=" * 60)
        
        return tests_passed == total_tests

def main():
    """Main function to run API tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test SMS Spam Detection API')
    parser.add_argument('--url', default='http://localhost:8000', help='API base URL')
    parser.add_argument('--model', default='naive_bayes', help='Model to test')
    parser.add_argument('--benchmark', type=int, default=50, help='Number of requests for benchmarking')
    
    args = parser.parse_args()
    
    tester = APITester(args.url)
    
    # Run comprehensive tests
    success = tester.run_all_tests()
    
    if success:
        print("🎉 All tests passed! API is working correctly.")
        exit(0)
    else:
        print("❌ Some tests failed. Check the API implementation.")
        exit(1)

if __name__ == "__main__":
    main()
