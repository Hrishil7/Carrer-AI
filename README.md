# 🧭 CareerAI - Personalized Career Companion

A modern, AI-powered career development platform with beautiful UI and comprehensive features.

## ✨ Features

- 🔍 **Ikigai Journaling** - Discover your purpose through AI-powered reflection
- 💡 **AI Project Ideas** - Generate personalized, resume-ready project suggestions
- 🤔 **Delta-4 Reflection** - Analyze friction vs delight in your career journey
- 📝 **LinkedIn Post Generator** - Create engaging professional content
- 🎯 **Career Coach** - Get personalized guidance and next steps
- 📊 **Progress Tracking** - Monitor your career development journey

## 🚀 Tech Stack

- **Backend**: FastAPI (Python)
- **Frontend**: Streamlit
- **AI**: Perplexity Sonar API
- **Database**: Supabase PostgreSQL / SQLite
- **Styling**: Modern CSS with glassmorphism

## 📦 Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd careerai
```

2. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

5. Run the application:
```bash
# Start backend
uvicorn backend:app --reload

# Start frontend (new terminal)
streamlit run ui_premium_v2.py
```

## 🎨 UI Preview

Modern, clean interface with:
- Indigo + Pink gradient theme
- Glassmorphism effects
- Smooth animations
- Professional typography (Inter font)

## 📚 API Documentation

See [API_REFERENCE.md](API_REFERENCE.md) for detailed API documentation.

## 🧪 Testing

```bash
pytest tests/
```

## 📝 License

MIT License

## 👤 Author

Built with ❤️ for career success
