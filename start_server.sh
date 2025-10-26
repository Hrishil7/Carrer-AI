#!/bin/bash
# Quick Start Script for CareerAI Backend

echo "🏆 CareerAI Backend - Quick Start"
echo "================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

# Check if uvicorn is available
if ! python3 -c "import uvicorn" &> /dev/null; then
    echo "📦 Installing uvicorn..."
    pip3 install uvicorn
fi

# Check if backend imports successfully
echo "🔍 Testing backend..."
if python3 -c "import backend" &> /dev/null; then
    echo "✅ Backend imports successfully!"
else
    echo "❌ Backend import failed. Please check your dependencies."
    exit 1
fi

# Start the server
echo "🚀 Starting CareerAI Backend Server..."
echo "📍 Server will be available at:"
echo "   - API: http://127.0.0.1:8000"
echo "   - Docs: http://127.0.0.1:8000/docs"
echo "   - Health: http://127.0.0.1:8000/health"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the server
python3 -m uvicorn backend:app --reload --host 127.0.0.1 --port 8000
