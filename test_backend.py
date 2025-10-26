#!/usr/bin/env python3
"""
CareerAI Backend - Final Test Script
This script tests all endpoints and shows you the expected outputs for UI development
"""

import requests
import json
import time

BASE_URL = "http://127.0.0.1:8000"

def test_endpoint(method, endpoint, data=None, headers=None):
    """Test an endpoint and return the response"""
    url = f"{BASE_URL}{endpoint}"
    try:
        if method.upper() == "GET":
            response = requests.get(url, headers=headers)
        elif method.upper() == "POST":
            response = requests.post(url, json=data, headers=headers)
        elif method.upper() == "PATCH":
            response = requests.patch(url, json=data, headers=headers)
        
        return {
            "status_code": response.status_code,
            "data": response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text
        }
    except Exception as e:
        return {"error": str(e)}

def main():
    print("🏆 CareerAI Backend - Final Test Results")
    print("=" * 50)
    
    # Test 1: Health Check
    print("\n1. 🏥 Health Check")
    print("-" * 20)
    result = test_endpoint("GET", "/health")
    print(f"Status: {result.get('status_code', 'Error')}")
    print(f"Response: {json.dumps(result.get('data', result.get('error', 'No data')), indent=2)}")
    
    # Test 2: Get Phases
    print("\n2. 📋 Get Phases")
    print("-" * 20)
    result = test_endpoint("GET", "/phases")
    print(f"Status: {result.get('status_code', 'Error')}")
    print(f"Response: {json.dumps(result.get('data', result.get('error', 'No data')), indent=2)}")
    
    # Test 3: Register User
    print("\n3. 👤 Register User")
    print("-" * 20)
    register_data = {
        "name": "Test User",
        "email": "test@example.com",
        "password": "testpassword123"
    }
    result = test_endpoint("POST", "/auth/register", register_data)
    print(f"Status: {result.get('status_code', 'Error')}")
    print(f"Response: {json.dumps(result.get('data', result.get('error', 'No data')), indent=2)}")
    
    # Extract token for authenticated requests
    token = None
    if result.get('status_code') == 200 and 'data' in result:
        token = result['data'].get('token')
    
    # Test 4: Login User
    print("\n4. 🔐 Login User")
    print("-" * 20)
    login_data = {
        "email": "test@example.com",
        "password": "testpassword123"
    }
    result = test_endpoint("POST", "/auth/login", login_data)
    print(f"Status: {result.get('status_code', 'Error')}")
    print(f"Response: {json.dumps(result.get('data', result.get('error', 'No data')), indent=2)}")
    
    if result.get('status_code') == 200 and 'data' in result:
        token = result['data'].get('token')
    
    # Test 5: Get Current User (Authenticated)
    if token:
        print("\n5. 👤 Get Current User (Authenticated)")
        print("-" * 20)
        headers = {"Authorization": f"Bearer {token}"}
        result = test_endpoint("GET", "/auth/me", headers=headers)
        print(f"Status: {result.get('status_code', 'Error')}")
        print(f"Response: {json.dumps(result.get('data', result.get('error', 'No data')), indent=2)}")
        
        # Test 6: Update Phase Progress
        print("\n6. 📈 Update Phase Progress")
        print("-" * 20)
        progress_data = {
            "current_phase": "Introspection",
            "progress": 50.0
        }
        result = test_endpoint("PATCH", "/phase/progress", progress_data, headers)
        print(f"Status: {result.get('status_code', 'Error')}")
        print(f"Response: {json.dumps(result.get('data', result.get('error', 'No data')), indent=2)}")
        
        # Test 7: Get Projects
        print("\n7. 📁 Get Projects")
        print("-" * 20)
        result = test_endpoint("GET", "/projects", headers=headers)
        print(f"Status: {result.get('status_code', 'Error')}")
        print(f"Response: {json.dumps(result.get('data', result.get('error', 'No data')), indent=2)}")
        
        # Test 8: Get Tasks
        print("\n8. ✅ Get Tasks")
        print("-" * 20)
        result = test_endpoint("GET", "/tasks")
        print(f"Status: {result.get('status_code', 'Error')}")
        print(f"Response: {json.dumps(result.get('data', result.get('error', 'No data')), indent=2)}")
        
        # Test 9: Get Analytics Metrics
        print("\n9. 📊 Get Analytics Metrics")
        print("-" * 20)
        result = test_endpoint("GET", "/analytics/metrics", headers=headers)
        print(f"Status: {result.get('status_code', 'Error')}")
        print(f"Response: {json.dumps(result.get('data', result.get('error', 'No data')), indent=2)}")
        
        # Test 10: Get User Insights
        print("\n10. 🧠 Get User Insights")
        print("-" * 20)
        result = test_endpoint("GET", "/ai/insights", headers=headers)
        print(f"Status: {result.get('status_code', 'Error')}")
        print(f"Response: {json.dumps(result.get('data', result.get('error', 'No data')), indent=2)}")
        
        # Test 11: Get Notifications
        print("\n11. 🔔 Get Notifications")
        print("-" * 20)
        result = test_endpoint("GET", "/notifications", headers=headers)
        print(f"Status: {result.get('status_code', 'Error')}")
        print(f"Response: {json.dumps(result.get('data', result.get('error', 'No data')), indent=2)}")
    
    # Test 12: Admin Health Check
    print("\n12. 🔧 Admin Health Check")
    print("-" * 20)
    result = test_endpoint("GET", "/admin/health")
    print(f"Status: {result.get('status_code', 'Error')}")
    print(f"Response: {json.dumps(result.get('data', result.get('error', 'No data')), indent=2)}")
    
    print("\n" + "=" * 50)
    print("🎉 Backend Testing Complete!")
    print("📚 API Documentation: http://127.0.0.1:8000/docs")
    print("🔍 ReDoc Documentation: http://127.0.0.1:8000/redoc")

if __name__ == "__main__":
    main()
