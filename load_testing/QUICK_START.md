# Quick Start Guide for Flood Testing

## Setup (One-time)

```bash
# 1. Install requirements
pip install locust requests

# 2. Make sure API is running
curl -s http://localhost:8000/health
```

## Run Flood Tests

### Option 1: Simple Python Script (Recommended)

```bash
# Quick test (30 seconds, 5 users)
python simple_flood_test.py quick

# Full test suite (3 different load levels)
python simple_flood_test.py

# Custom stress test
python simple_flood_test.py stress
```

### Option 2: Direct Locust Commands

```bash
# Quick test
locust -f locustfile.py --host=http://localhost:8000 --users=5 --spawn-rate=1 --run-time=30s --headless

# Medium load test
locust -f locustfile.py --host=http://localhost:8000 --users=15 --spawn-rate=3 --run-time=60s --headless

# Interactive mode (web interface at http://localhost:8089)
locust -f locustfile.py --host=http://localhost:8000
```

## Files in this directory:

- `locustfile.py` - Main flood testing script
- `simple_flood_test.py` - Easy-to-use test runner
- `load_testing_requirements.txt` - Required packages
- `README.md` - Comprehensive results and documentation
- `*.json` - Test results files

## Quick Results Summary:

**100% Success Rate** under flood testing  
**0.634s Average Response Time**  
**Performance Rating: EXCELLENT**

See README.md for complete analysis and results.
