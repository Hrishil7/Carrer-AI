#!/bin/bash
cd "/Users/hrishilshah/backend+frontend hackathon"

echo "🚀 Pushing to GitHub..."
git remote add origin https://github.com/Hrishil7/Carrer-AI.git 2>/dev/null || echo "Remote already exists"
git branch -M main
git push -u origin main

echo ""
echo "✅ Done! Your code is now on GitHub!"
echo "View it at: https://github.com/Hrishil7/Carrer-AI"
