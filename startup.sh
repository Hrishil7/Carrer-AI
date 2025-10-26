#!/bin/bash
# startup.sh - CareerAI Backend Startup Script

set -e

echo "🚀 Starting CareerAI Backend..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip3 first."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found. Creating from template..."
    cp .env.example .env 2>/dev/null || echo "Please create .env file with your configuration"
fi

# Check if DATABASE_URL is set
if ! grep -q "DATABASE_URL=" .env || grep -q "your_database_url_here" .env; then
    echo "⚠️  Please set DATABASE_URL in .env file"
fi

# Check if OPENAI_API_KEY is set
if ! grep -q "OPENAI_API_KEY=" .env || grep -q "your_openai_api_key_here" .env; then
    echo "⚠️  Please set OPENAI_API_KEY in .env file"
fi

# Check if JWT_SECRET_KEY is set
if ! grep -q "JWT_SECRET_KEY=" .env || grep -q "your_jwt_secret_key_here" .env; then
    echo "⚠️  Please set JWT_SECRET_KEY in .env file"
fi

echo "✅ Setup complete!"
echo ""
echo "🔧 To start the server:"
echo "   source venv/bin/activate"
echo "   uvicorn backend:app --reload"
echo ""
echo "📚 API Documentation will be available at:"
echo "   http://localhost:8000/docs"
echo ""
echo "🏥 Health check:"
echo "   http://localhost:8000/health"
