#!/usr/bin/env python3
"""
SMS Spam Detection API Demo

This script demonstrates the real-time SMS spam detection API capabilities.
"""

import asyncio
import time
import requests
from datetime import datetime
import json

# Demo SMS messages
DEMO_MESSAGES = [
    {
        "text": "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question",
        "expected": "spam",
        "description": "Classic spam message with promotional content"
    },
    {
        "text": "Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat",
        "expected": "ham",
        "description": "Legitimate SMS with casual language"
    },
    {
        "text": "WINNER!! As a valued network customer you have been selected to receivea £900 prize reward! To claim call 09061701461",
        "expected": "spam",
        "description": "Prize scam message"
    },
    {
        "text": "I've been searching for the right words to thank you for this breather. I promise i wont take your help for granted",
        "expected": "ham",
        "description": "Personal message of gratitude"
    },
    {
        "text": "URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot! Txt the word: CLAIM to No: 81010",
        "expected": "spam",
        "description": "Urgent prize notification spam"
    },
    {
        "text": "Ok lar... Joking wif u oni...",
        "expected": "ham",
        "description": "Casual conversation"
    }
]

class APIDemo:
    """Demo class for SMS Spam Detection API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize demo.
        
        Args:
            base_url: Base URL of the API
        """
        self.base_url = base_url
        self.session = requests.Session()
        
    def check_api_status(self) -> bool:
        """Check if API is running and healthy."""
        print("🔍 Checking API status...")
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ API is healthy!")
                print(f"   Status: {data['status']}")
                print(f"   Models loaded: {data['models_loaded']}/{data['total_models']}")
                print(f"   Uptime: {data['uptime_seconds']:.1f} seconds")
                return True
            else:
                print(f"❌ API health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Cannot connect to API: {e}")
            print("   Make sure the API is running: python sms_api.py")
            return False
    
    def demo_single_predictions(self):
        """Demonstrate single message predictions."""
        print("\n" + "="*60)
        print("📱 SINGLE MESSAGE PREDICTION DEMO")
        print("="*60)
        
        for i, msg in enumerate(DEMO_MESSAGES, 1):
            print(f"\n📨 Message {i}: {msg['description']}")
            print(f"Text: {msg['text'][:80]}{'...' if len(msg['text']) > 80 else ''}")
            print(f"Expected: {msg['expected']}")
            
            try:
                # Test with different models
                models = ['naive_bayes', 'logistic_regression', 'random_forest']
                predictions = {}
                
                for model in models:
                    response = self.session.post(
                        f"{self.base_url}/predict",
                        params={"model": model},
                        json={"text": msg['text']},
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        predictions[model] = {
                            'prediction': data['prediction'],
                            'probability': data['probability'],
                            'time': data['processing_time_ms']
                        }
                    else:
                        print(f"   ❌ {model}: API error {response.status_code}")
                
                # Display results
                print("   Results:")
                for model, result in predictions.items():
                    correct = "✅" if result['prediction'] == msg['expected'] else "❌"
                    print(f"   {correct} {model}: {result['prediction']} "
                          f"({result['probability']:.3f}) - {result['time']:.1f}ms")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
    
    def demo_batch_predictions(self):
        """Demonstrate batch message predictions."""
        print("\n" + "="*60)
        print("📦 BATCH PREDICTION DEMO")
        print("="*60)
        
        # Prepare batch
        batch_messages = {
            "messages": [
                {"text": msg["text"], "sender_id": f"sender_{i}"}
                for i, msg in enumerate(DEMO_MESSAGES[:4])
            ]
        }
        
        print(f"Processing {len(batch_messages['messages'])} messages in batch...")
        
        try:
            response = self.session.post(
                f"{self.base_url}/predict/batch",
                params={"model": "naive_bayes"},
                json=batch_messages,
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Batch processing completed!")
                print(f"   Total messages: {data['total_messages']}")
                print(f"   Spam detected: {data['spam_count']}")
                print(f"   Ham detected: {data['ham_count']}")
                print(f"   Total processing time: {data['total_processing_time_ms']:.1f}ms")
                print(f"   Average time per message: {data['total_processing_time_ms']/data['total_messages']:.1f}ms")
                
                print("\n   Individual results:")
                for i, pred in enumerate(data['predictions']):
                    expected = DEMO_MESSAGES[i]['expected']
                    correct = "✅" if pred['prediction'] == expected else "❌"
                    print(f"   {correct} Message {i+1}: {pred['prediction']} "
                          f"({pred['probability']:.3f}) - Expected: {expected}")
            else:
                print(f"❌ Batch prediction failed: {response.status_code}")
                print(f"   Response: {response.text}")
                
        except Exception as e:
            print(f"❌ Batch prediction error: {e}")
    
    def demo_ensemble_predictions(self):
        """Demonstrate ensemble predictions."""
        print("\n" + "="*60)
        print("🎯 ENSEMBLE PREDICTION DEMO")
        print("="*60)
        
        test_message = DEMO_MESSAGES[0]  # Use first spam message
        
        print(f"Testing ensemble prediction on: {test_message['description']}")
        print(f"Text: {test_message['text'][:80]}...")
        print(f"Expected: {test_message['expected']}")
        
        try:
            response = self.session.post(
                f"{self.base_url}/predict/ensemble",
                json={"text": test_message['text']},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                correct = "✅" if data['prediction'] == test_message['expected'] else "❌"
                print(f"✅ Ensemble prediction completed!")
                print(f"   {correct} Prediction: {data['prediction']}")
                print(f"   Confidence: {data['probability']:.3f}")
                print(f"   Models used: {data['model_used']}")
                print(f"   Processing time: {data['processing_time_ms']:.1f}ms")
            else:
                print(f"❌ Ensemble prediction failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Ensemble prediction error: {e}")
    
    def demo_performance_benchmark(self, num_requests: int = 20):
        """Demonstrate API performance."""
        print("\n" + "="*60)
        print("⚡ PERFORMANCE BENCHMARK DEMO")
        print("="*60)
        
        test_message = {"text": "Free entry in 2 a wkly comp to win FA Cup final tkts"}
        response_times = []
        successful_requests = 0
        
        print(f"Running {num_requests} requests...")
        start_time = time.time()
        
        for i in range(num_requests):
            try:
                request_start = time.time()
                response = self.session.post(
                    f"{self.base_url}/predict",
                    params={"model": "naive_bayes"},
                    json=test_message,
                    timeout=5
                )
                request_end = time.time()
                
                if response.status_code == 200:
                    successful_requests += 1
                    response_times.append((request_end - request_start) * 1000)
                
                if (i + 1) % 5 == 0:
                    print(f"   Completed {i + 1}/{num_requests} requests...")
                    
            except Exception as e:
                print(f"   Request {i + 1} failed: {e}")
        
        total_time = time.time() - start_time
        
        if response_times:
            print(f"\n✅ Performance benchmark completed!")
            print(f"   Success rate: {successful_requests}/{num_requests} ({successful_requests/num_requests:.1%})")
            print(f"   Total time: {total_time:.2f} seconds")
            print(f"   Requests per second: {num_requests/total_time:.1f}")
            print(f"   Average response time: {sum(response_times)/len(response_times):.1f}ms")
            print(f"   Min response time: {min(response_times):.1f}ms")
            print(f"   Max response time: {max(response_times):.1f}ms")
        else:
            print("❌ No successful requests for benchmarking")
    
    def run_complete_demo(self):
        """Run the complete API demonstration."""
        print("🚀 SMS SPAM DETECTION API - COMPLETE DEMO")
        print("=" * 60)
        
        # Check API status
        if not self.check_api_status():
            return
        
        # Run all demos
        self.demo_single_predictions()
        self.demo_batch_predictions()
        self.demo_ensemble_predictions()
        self.demo_performance_benchmark()
        
        print("\n" + "="*60)
        print("🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("📚 API Documentation: http://localhost:8000/docs")
        print("🔍 Health Check: http://localhost:8000/health")
        print("📊 Model Info: http://localhost:8000/models")

def main():
    """Main function to run the demo."""
    import argparse
    
    parser = argparse.ArgumentParser(description='SMS Spam Detection API Demo')
    parser.add_argument('--url', default='http://localhost:8000', help='API base URL')
    parser.add_argument('--requests', type=int, default=20, help='Number of requests for benchmark')
    
    args = parser.parse_args()
    
    demo = APIDemo(args.url)
    demo.run_complete_demo()

if __name__ == "__main__":
    main()
