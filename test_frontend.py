#!/usr/bin/env python3
"""
CareerAI Frontend Test Script
This script tests the frontend functionality by simulating user interactions
"""

import requests
import json
import time
import random

BASE_URL = "http://127.0.0.1:8000"
FRONTEND_URL = "http://localhost:8501"

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

def create_test_user():
    """Create a test user and return auth token"""
    test_email = f"testuser{random.randint(1000, 9999)}@example.com"
    register_data = {
        "name": "Test User",
        "email": test_email,
        "password": "testpassword123"
    }
    
    result = test_endpoint("POST", "/auth/register", register_data)
    if result.get('status_code') == 200:
        return result['data']['token'], test_email
    else:
        # Try to login if user already exists
        login_data = {
            "email": test_email,
            "password": "testpassword123"
        }
        result = test_endpoint("POST", "/auth/login", login_data)
        if result.get('status_code') == 200:
            return result['data']['token'], test_email
        else:
            return None, None

def test_frontend_features():
    """Test all frontend features with sample data"""
    print("🎨 CareerAI Frontend Test Results")
    print("=" * 50)
    
    # Create test user
    print("\n1. 👤 Creating Test User")
    print("-" * 20)
    token, email = create_test_user()
    if not token:
        print("❌ Failed to create/login test user")
        return
    
    headers = {"Authorization": f"Bearer {token}"}
    print(f"✅ Test user created: {email}")
    
    # Test Dashboard Data
    print("\n2. 📊 Dashboard Data")
    print("-" * 20)
    
    # Get user profile
    result = test_endpoint("GET", "/auth/me", headers=headers)
    if result.get('status_code') == 200:
        user_data = result['data']
        print(f"✅ User Profile: {user_data['name']} - {user_data['email']}")
        print(f"   Current Phase: {user_data.get('current_phase', 'Not set')}")
        print(f"   Progress: {user_data.get('progress', 0)}%")
    else:
        print(f"❌ Failed to get user profile: {result}")
    
    # Get analytics
    result = test_endpoint("GET", "/analytics/metrics", headers=headers)
    if result.get('status_code') == 200:
        analytics = result['data']
        print(f"✅ Analytics: {analytics['total_events']} total events")
        print(f"   Activity Score: {analytics['activity_score']}")
    else:
        print(f"❌ Failed to get analytics: {result}")
    
    # Get insights
    result = test_endpoint("GET", "/ai/insights", headers=headers)
    if result.get('status_code') == 200:
        insights = result['data']
        print(f"✅ Insights: Productivity {insights['productivity_score']}%")
        print(f"   Growth Areas: {len(insights['growth_areas'])}")
    else:
        print(f"❌ Failed to get insights: {result}")
    
    # Test Introspection Tab
    print("\n3. 🔍 Introspection Tab (Ikigai)")
    print("-" * 20)
    
    sample_journal = """
    I love teaching and helping others learn complex concepts. I'm passionate about making education more accessible through technology. 
    I enjoy working with AI and machine learning, and I'm good at breaking down technical concepts into simple explanations. 
    I want to make a positive impact on people's lives by helping them develop new skills and advance their careers. 
    I can be paid for creating educational content, consulting, and building learning platforms.
    """
    
    ikigai_data = {"journal_text": sample_journal}
    result = test_endpoint("POST", "/ai/ikigai", ikigai_data, headers)
    if result.get('status_code') == 200:
        ikigai_result = result['data']
        print(f"✅ Ikigai Analysis Complete")
        print(f"   Journal Entry ID: {ikigai_result['journal_entry_id']}")
        print(f"   AI Summary: {ikigai_result['ai_summary'][:100]}...")
        print(f"   Sentiment Score: {ikigai_result['sentiment_score']}")
    else:
        print(f"❌ Failed ikigai analysis: {result}")
    
    # Test Exploration Tab
    print("\n4. 💡 Exploration Tab (Project Ideas)")
    print("-" * 20)
    
    ikigai_summary = "I love teaching with AI, I'm good at explaining complex concepts, I want to make education accessible, and I can be paid for creating educational content."
    project_data = {"ikigai_summary": ikigai_summary}
    result = test_endpoint("POST", "/ai/project-ideas", project_data, headers)
    if result.get('status_code') == 200:
        projects = result['data']
        print(f"✅ Generated {len(projects)} project ideas:")
        for i, project in enumerate(projects[:3], 1):
            print(f"   {i}. {project['title']}")
            print(f"      {project['description'][:80]}...")
    else:
        print(f"❌ Failed to generate project ideas: {result}")
    
    # Test Reflection Tab
    print("\n5. 🤔 Reflection Tab (Delta-4)")
    print("-" * 20)
    
    reflection_data = {
        "friction": "I'm struggling with time management and feeling overwhelmed by too many projects. I'm not sure which skills to prioritize for my career growth.",
        "delight": "I love when I can help someone understand a complex concept. I feel energized when working on AI projects and I'm proud of my recent teaching achievements."
    }
    result = test_endpoint("POST", "/ai/reflection", reflection_data, headers)
    if result.get('status_code') == 200:
        reflection_result = result['data']
        print(f"✅ Reflection Analysis Complete")
        print(f"   Reflection ID: {reflection_result['reflection_id']}")
        print(f"   AI Summary: {reflection_result['ai_summary'][:100]}...")
    else:
        print(f"❌ Failed reflection analysis: {result}")
    
    # Test Action Tab
    print("\n6. 📈 Action Tab (Phase Progress)")
    print("-" * 20)
    
    progress_data = {
        "current_phase": "Exploration",
        "progress": 75.0
    }
    result = test_endpoint("PATCH", "/phase/progress", progress_data, headers)
    if result.get('status_code') == 200:
        progress_result = result['data']
        print(f"✅ Phase Progress Updated")
        print(f"   Current Phase: {progress_result['current_phase']}")
        print(f"   Progress: {progress_result['progress']}%")
    else:
        print(f"❌ Failed to update progress: {result}")
    
    # Test Projects Tab
    print("\n7. 🚀 Projects Tab")
    print("-" * 20)
    
    result = test_endpoint("GET", "/projects", headers=headers)
    if result.get('status_code') == 200:
        projects_data = result['data']
        projects = projects_data.get('projects', [])
        print(f"✅ Found {len(projects)} projects")
        for project in projects[:3]:
            print(f"   - {project['title']} ({project['status']})")
    else:
        print(f"❌ Failed to get projects: {result}")
    
    # Test Coach Tab
    print("\n8. 🎯 Coach Tab (Career Guidance)")
    print("-" * 20)
    
    guidance_data = {
        "current_role_or_stage": "Junior AI Engineer",
        "goals": "I want to become a senior AI engineer, lead technical teams, and contribute to open-source projects that make a real impact.",
        "recent_work_summary": "Built a small RAG app with LangChain, contributed to an open-source project, completed a machine learning course."
    }
    result = test_endpoint("POST", "/ai/guidance", guidance_data, headers)
    if result.get('status_code') == 200:
        guidance_result = result['data']
        print(f"✅ Career Guidance Generated")
        print(f"   Advice: {guidance_result['advice'][:100]}...")
        print(f"   Next Steps: {len(guidance_result['next_steps'])} recommendations")
    else:
        print(f"❌ Failed to get guidance: {result}")
    
    # Test Notifications
    print("\n9. 🔔 Notifications")
    print("-" * 20)
    
    result = test_endpoint("GET", "/notifications", headers=headers)
    if result.get('status_code') == 200:
        notifications = result['data']
        print(f"✅ Found {len(notifications)} notifications")
    else:
        print(f"❌ Failed to get notifications: {result}")
    
    # Test Performance Stats
    print("\n10. 📊 Performance Stats")
    print("-" * 20)
    
    result = test_endpoint("GET", "/admin/performance")
    if result.get('status_code') == 200:
        perf_data = result['data']
        print(f"✅ Performance Stats Available")
        print(f"   Endpoints tracked: {len(perf_data.get('endpoints', {}))}")
        print(f"   Cache items: {perf_data.get('cache_stats', {}).get('total_cached_items', 0)}")
    else:
        print(f"❌ Failed to get performance stats: {result}")
    
    print("\n" + "=" * 50)
    print("🎉 Frontend Test Complete!")
    print("📱 Frontend URL: http://localhost:8501")
    print("🔧 Backend API: http://127.0.0.1:8000")
    print("📚 API Docs: http://127.0.0.1:8000/docs")

if __name__ == "__main__":
    test_frontend_features()
