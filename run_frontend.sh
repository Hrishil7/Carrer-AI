#!/bin/bash

# CareerAI Frontend Startup Script
echo "🚀 Starting CareerAI Frontend..."

# Check if virtual environment exists
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

# Check if backend is running
echo "🔍 Checking backend connection..."
if curl -s http://127.0.0.1:8000/health > /dev/null; then
    echo "✅ Backend is running"
else
    echo "⚠️  Backend not detected. Please start the backend first:"
    echo "   cd backend && python backend.py"
    echo "   or run: ./start_server.sh"
fi

# Start Streamlit
echo "🎨 Starting Streamlit frontend..."
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
