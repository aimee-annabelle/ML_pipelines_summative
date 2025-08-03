"""
Simplified Locust Load Testing for Pneumonia Detection API

This script simulates flood testing by sending rapid requests to the prediction endpoint.
It measures response times and throughput under different load conditions.

Usage:
  # Quick test (5 users, 30 seconds)
  locust -f locustfile.py --host=http://localhost:8000 --users=5 --spawn-rate=1 --run-time=30s --headless

  # Stress test (20 users, 2 minutes) 
  locust -f locustfile.py --host=http://localhost:8000 --users=20 --spawn-rate=5 --run-time=120s --headless

  # Interactive mode
  locust -f locustfile.py --host=http://localhost:8000
"""

import os
import random
import time
import json
import logging
from locust import HttpUser, task, between, events

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for metrics tracking
request_metrics = {
    'response_times': [],
    'success_count': 0,
    'error_count': 0,
    'predictions': []
}

class FloodTestUser(HttpUser):
    """Simulates flood testing of the Pneumonia Detection API"""
    
    # Aggressive timing for flood testing
    wait_time = between(0.1, 2)
    
    
    def on_start(self):
        """Initialize user with test images"""
        self.test_images = self.load_test_images()
        logger.info(f"User started with {len(self.test_images)} test images")
    
    def load_test_images(self):
        """Load available test images for flood testing"""
        test_images = []
        
        # Look for test images in data/test directory
        test_paths = [
            "../data/test/NORMAL",
            "../data/test/PNEUMONIA",
            "data/test/NORMAL", 
            "data/test/PNEUMONIA"
        ]
        
        for path in test_paths:
            if os.path.exists(path):
                files = [f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                label = "NORMAL" if "NORMAL" in path else "PNEUMONIA"
                for file in files[:3]:  # Limit to 3 per category for testing
                    test_images.append((os.path.join(path, file), label))
        
        if not test_images:
            # Fallback dummy data if no images found
            test_images = [("dummy_normal.jpg", "NORMAL"), ("dummy_pneumonia.jpg", "PNEUMONIA")]
        
        return test_images
    
    @task(90)  # 90% of requests are predictions (flood testing focus)
    def flood_predict(self):
        """Main flood testing task - rapid prediction requests"""
        if not self.test_images:
            return
        
        image_path, expected_label = random.choice(self.test_images)
        
        start_time = time.time()
        try:
            if os.path.exists(image_path):
                with open(image_path, 'rb') as f:
                    files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
                    response = self.client.post("/api/v1/predict", files=files, timeout=10)
            else:
                # Use dummy data for testing
                files = {'file': ('test.jpg', b'dummy_image_data', 'image/jpeg')}
                response = self.client.post("/api/v1/predict", files=files, timeout=10)
            
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                prediction = data.get('prediction', 'unknown')
                confidence = data.get('confidence', 0)
                
                # Track metrics
                request_metrics['response_times'].append(response_time)
                request_metrics['success_count'] += 1
                request_metrics['predictions'].append({
                    'prediction': prediction,
                    'confidence': confidence,
                    'response_time': response_time
                })
                
                logger.info(f"✓ Prediction: {prediction}, Confidence: {confidence:.2f}, Time: {response_time:.2f}s")
            else:
                request_metrics['error_count'] += 1
                logger.error(f"✗ Prediction failed: {response.status_code}")
                
        except Exception as e:
            request_metrics['error_count'] += 1
            logger.error(f"✗ Request exception: {e}")
    
    @task(8)  # 8% health checks
    def health_check(self):
        """Quick health monitoring"""
        try:
            response = self.client.get("/health", timeout=5)
            if response.status_code == 200:
                logger.debug("Health check: OK")
        except Exception as e:
            logger.error(f"Health check failed: {e}")
    
    @task(2)  # 2% status checks
    def status_check(self):
        """System status monitoring"""
        try:
            response = self.client.get("/api/v1/status", timeout=5)
            if response.status_code == 200:
                data = response.json()
                logger.debug(f"Status: {data.get('total_predictions', 0)} total predictions")
        except Exception as e:
            logger.error(f"Status check failed: {e}")


# Event handlers for test lifecycle
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Initialize flood test"""
    logger.info("=== FLOOD TEST STARTED ===")
    logger.info(f"Target: {environment.host}")
    logger.info(f"Users: {getattr(environment.runner, 'target_user_count', 'N/A')}")
    
    # Reset metrics
    request_metrics['response_times'] = []
    request_metrics['success_count'] = 0
    request_metrics['error_count'] = 0
    request_metrics['predictions'] = []


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Generate flood test summary"""
    logger.info("=== FLOOD TEST COMPLETED ===")
    
    if request_metrics['response_times']:
        response_times = request_metrics['response_times']
        
        # Calculate statistics
        avg_time = sum(response_times) / len(response_times)
        max_time = max(response_times)
        min_time = min(response_times)
        total_requests = len(response_times)
        
        # Calculate percentiles
        sorted_times = sorted(response_times)
        p50 = sorted_times[int(len(sorted_times) * 0.5)]
        p95 = sorted_times[int(len(sorted_times) * 0.95)]
        p99 = sorted_times[int(len(sorted_times) * 0.99)]
        
        # Print summary
        logger.info(f"FLOOD TEST RESULTS:")
        logger.info(f"  Total Prediction Requests: {total_requests}")
        logger.info(f"  Successful Requests: {request_metrics['success_count']}")
        logger.info(f"  Failed Requests: {request_metrics['error_count']}")
        logger.info(f"  Success Rate: {(request_metrics['success_count'] / max(total_requests, 1)) * 100:.1f}%")
        logger.info(f"  Average Response Time: {avg_time:.3f}s")
        logger.info(f"  Min Response Time: {min_time:.3f}s")
        logger.info(f"  Max Response Time: {max_time:.3f}s")
        logger.info(f"  50th Percentile: {p50:.3f}s")
        logger.info(f"  95th Percentile: {p95:.3f}s")
        logger.info(f"  99th Percentile: {p99:.3f}s")
        
        # Save results to JSON
        results = {
            "flood_test_summary": {
                "total_prediction_requests": total_requests,
                "successful_requests": request_metrics['success_count'],
                "failed_requests": request_metrics['error_count'],
                "success_rate_percent": (request_metrics['success_count'] / max(total_requests, 1)) * 100,
                "response_time_stats": {
                    "average_seconds": avg_time,
                    "min_seconds": min_time,
                    "max_seconds": max_time,
                    "p50_seconds": p50,
                    "p95_seconds": p95,
                    "p99_seconds": p99
                },
                "all_response_times": response_times,
                "sample_predictions": request_metrics['predictions'][:10]  # First 10 predictions
            }
        }
        
        with open("flood_test_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info("Results saved to flood_test_results.json")
    else:
        logger.warning("No prediction requests were made during the test")


@events.request.add_listener
def on_request(request_type, name, response_time, response_length, response, context, exception, **kwargs):
    """Track individual requests"""
    if exception:
        logger.debug(f"Request failed: {request_type} {name} - {exception}")
    elif response_time > 5000:  # Log slow requests (>5s)
        logger.warning(f"Slow request: {request_type} {name} - {response_time}ms")
