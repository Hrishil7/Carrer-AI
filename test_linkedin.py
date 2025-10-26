#!/usr/bin/env python3
"""
Test LinkedIn Post Generator
"""

import requests
import json
import time

BASE_URL = "http://127.0.0.1:8000"

def test_linkedin_post_generator():
    """Test the LinkedIn post generator with different scenarios"""
    
    print("🧪 Testing LinkedIn Post Generator")
    print("=" * 50)
    
    test_cases = [
        {
            "name": "Project Showcase",
            "data": {
                "post_type": "project",
                "content": "I just completed my first machine learning project using Python and scikit-learn. It was challenging but I learned so much about data preprocessing, model training, and evaluation. The project achieved 85% accuracy on the test dataset.",
                "tone": "professional",
                "include_hashtags": True,
                "include_call_to_action": True
            }
        },
        {
            "name": "Achievement Celebration",
            "data": {
                "post_type": "achievement",
                "content": "I just got promoted to Senior Software Engineer! After 2 years of hard work, learning new technologies, and leading several successful projects, I am excited to take on this new role.",
                "tone": "inspiring",
                "include_hashtags": True,
                "include_call_to_action": True
            }
        },
        {
            "name": "Learning Experience",
            "data": {
                "post_type": "learning",
                "content": "Just finished a comprehensive course on React and TypeScript. The combination of these technologies has completely changed how I approach frontend development. The type safety and component reusability are game-changers!",
                "tone": "casual",
                "include_hashtags": True,
                "include_call_to_action": True
            }
        },
        {
            "name": "Career Update",
            "data": {
                "post_type": "career_update",
                "content": "Excited to announce that I'm starting a new role as a Data Scientist at TechCorp! I'll be working on machine learning models to improve customer experience. Looking forward to this new chapter!",
                "tone": "personal",
                "include_hashtags": True,
                "include_call_to_action": True
            }
        },
        {
            "name": "Thought Leadership",
            "data": {
                "post_type": "thought_leadership",
                "content": "The future of AI isn't just about building smarter models—it's about building AI that understands context, ethics, and human values. As we advance, we must prioritize responsible AI development.",
                "tone": "technical",
                "include_hashtags": True,
                "include_call_to_action": True
            }
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['name']}")
        print("-" * 30)
        
        try:
            response = requests.post(
                f"{BASE_URL}/ai/linkedin-post",
                json=test_case["data"],
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                
                print(f"✅ Post Type: {test_case['data']['post_type']}")
                print(f"✅ Tone: {test_case['data']['tone']}")
                print(f"✅ Character Count: {data.get('character_count', 'N/A')}")
                print(f"✅ Hashtags: {len(data.get('hashtags', []))}")
                print(f"✅ Call-to-Action: {'Yes' if data.get('call_to_action') else 'No'}")
                print(f"✅ Engagement Tips: {len(data.get('engagement_tips', []))}")
                
                # Show a preview of the generated content
                post_content = data.get('post_content', '')
                preview = post_content[:100] + "..." if len(post_content) > 100 else post_content
                print(f"📝 Preview: {preview}")
                
            else:
                print(f"❌ Failed: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"❌ Error: {str(e)}")
        
        time.sleep(1)  # Rate limiting
    
    print("\n" + "=" * 50)
    print("🎉 LinkedIn Post Generator Test Complete!")
    print(f"📱 Frontend URL: http://localhost:8501")
    print(f"🔧 Backend API: {BASE_URL}")
    print(f"📚 API Docs: {BASE_URL}/docs")

if __name__ == "__main__":
    test_linkedin_post_generator()
