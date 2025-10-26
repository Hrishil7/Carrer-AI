#!/usr/bin/env python3
"""
Test Perplexity Sonar Integration
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Perplexity configuration
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
PERPLEXITY_MODEL = os.getenv("PERPLEXITY_MODEL", "sonar-medium-chat")
PERPLEXITY_BASE_URL = os.getenv("PERPLEXITY_BASE_URL", "https://api.perplexity.ai")

print("🧪 Testing Perplexity Sonar Integration")
print("=" * 50)

print(f"API Key: {PERPLEXITY_API_KEY[:10]}..." if PERPLEXITY_API_KEY else "❌ No API Key")
print(f"Model: {PERPLEXITY_MODEL}")
print(f"Base URL: {PERPLEXITY_BASE_URL}")

if not PERPLEXITY_API_KEY:
    print("❌ PERPLEXITY_API_KEY not found in environment")
    exit(1)

# Initialize client
try:
    client = OpenAI(
        api_key=PERPLEXITY_API_KEY,
        base_url=PERPLEXITY_BASE_URL
    )
    print("✅ Perplexity client initialized")
except Exception as e:
    print(f"❌ Failed to initialize client: {e}")
    exit(1)

# Test API call
try:
    print("\n🧠 Testing AI call...")
    response = client.chat.completions.create(
        model=PERPLEXITY_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful career coach."},
            {"role": "user", "content": "Give me a brief career advice in one sentence."}
        ],
        max_tokens=100
    )
    
    result = response.choices[0].message.content
    print(f"✅ AI Response: {result}")
    print("\n🎉 Perplexity Sonar integration working perfectly!")
    
except Exception as e:
    print(f"❌ API call failed: {e}")
    print(f"Error type: {type(e).__name__}")
    if hasattr(e, 'response'):
        print(f"Response: {e.response}")
