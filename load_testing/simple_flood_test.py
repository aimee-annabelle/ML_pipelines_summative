"""
Simple script to run flood tests and display results
"""

import subprocess
import json
import os
import sys

def run_flood_test(users=10, duration=60, host="http://localhost:8000"):
    """Run a flood test with specified parameters"""
    
    print(f"Starting flood test:")
    print(f"   Users: {users}")
    print(f"   Duration: {duration} seconds")
    print(f"   Target: {host}")
    print("-" * 50)
    
    # Run locust command
    cmd = [
        "locust", 
        "-f", "locustfile.py",
        "--host", host,
        "--users", str(users),
        "--spawn-rate", str(max(1, users // 5)),  # Spawn rate is 1/5 of users
        "--run-time", f"{duration}s",
        "--headless"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=duration + 30)
        
        if result.returncode == 0:
            print("Flood test completed successfully!")
        else:
            print("Flood test failed!")
            print(f"Error: {result.stderr}")
            return None
            
    except subprocess.TimeoutExpired:
        print("Test timed out")
        return None
    except Exception as e:
        print(f"Error running test: {e}")
        return None
    
    # Read results if available
    if os.path.exists("flood_test_results.json"):
        with open("flood_test_results.json", "r") as f:
            results = json.load(f)
        return results
    
    return None

def display_results(results):
    """Display flood test results in a readable format"""
    if not results or "flood_test_summary" not in results:
        print("No results available")
        return
    
    summary = results["flood_test_summary"]
    
    print("\nFLOOD TEST RESULTS")
    print("=" * 50)
    print(f"Total Requests: {summary['total_prediction_requests']}")
    print(f"Successful: {summary['successful_requests']}")
    print(f"Failed: {summary['failed_requests']}")
    print(f"Success Rate: {summary['success_rate_percent']:.1f}%")
    print()
    
    print("RESPONSE TIME ANALYSIS")
    print("-" * 30)
    stats = summary["response_time_stats"]
    print(f"Average: {stats['average_seconds']:.3f}s")
    print(f"Minimum: {stats['min_seconds']:.3f}s")
    print(f"Maximum: {stats['max_seconds']:.3f}s")
    print(f"95th Percentile: {stats['p95_seconds']:.3f}s")
    print(f"99th Percentile: {stats['p99_seconds']:.3f}s")
    
    # Performance assessment
    avg_time = stats['average_seconds']
    if avg_time < 1.0:
        performance = "EXCELLENT"
    elif avg_time < 2.0:
        performance = "GOOD"
    elif avg_time < 5.0:
        performance = "FAIR"
    else:
        performance = "POOR"
    
    print(f"\nPerformance Rating: {performance}")
    print("=" * 50)

def main():
    """Main function to run flood tests"""
    
    # Default test parameters
    tests = [
        {"name": "Light Load", "users": 5, "duration": 30},
        {"name": "Medium Load", "users": 15, "duration": 60},
        {"name": "Heavy Load", "users": 25, "duration": 90}
    ]
    
    if len(sys.argv) > 1:
        test_type = sys.argv[1].lower()
        if test_type == "quick":
            tests = [{"name": "Quick Test", "users": 5, "duration": 30}]
        elif test_type == "stress":
            tests = [{"name": "Stress Test", "users": 50, "duration": 120}]
    
    all_results = []
    
    for test in tests:
        print(f"\nRunning {test['name']} Test...")
        results = run_flood_test(
            users=test["users"], 
            duration=test["duration"]
        )
        
        if results:
            print(f"\n{test['name']} Results:")
            display_results(results)
            all_results.append({
                "test_name": test["name"],
                "parameters": test,
                "results": results
            })
        
        print("\n" + "="*60 + "\n")
    
    # Save all results
    if all_results:
        with open("all_flood_test_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        print("All results saved to all_flood_test_results.json")

if __name__ == "__main__":
    main()
