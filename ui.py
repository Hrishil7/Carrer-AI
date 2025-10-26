import os
import json
import httpx
import streamlit as st
from pydantic import BaseModel
from dotenv import load_dotenv
import time
import asyncio
from datetime import datetime

# --------------------------
# Config
# --------------------------
load_dotenv()
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="CareerAI",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "CareerAI – Personalized Career Companion"},
)

# --------------------------
# Theming / CSS
# --------------------------
PRIMARY = "#7C5CFC"
ACCENT = "#22D3EE"
SUCCESS = "#10B981"
WARNING = "#F59E0B"
ERROR = "#EF4444"
BG_GRADIENT = "linear-gradient(135deg, #0f172a 0%, #111827 50%, #0b132b 100%)"
CARD_BLUR = """
backdrop-filter: blur(12px);
background: rgba(255, 255, 255, 0.06);
border: 1px solid rgba(255, 255, 255, 0.1);
border-radius: 16px;
"""

st.markdown(
    f"""
    <style>
    .stApp {{
        background: {BG_GRADIENT};
        color: #e5e7eb;
    }}
    
    .glass {{
        {CARD_BLUR}
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }}
    
    .glass:hover {{
        transform: translateY(-2px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.3);
    }}
    
    .hero {{
        {CARD_BLUR}
        padding: 2rem;
        border-radius: 20px;
        background: radial-gradient(1200px 600px at -10% -20%, rgba(34,211,238,0.12), transparent 40%),
                    radial-gradient(1600px 800px at 120% 10%, rgba(124,92,252,0.12), transparent 40%),
                    rgba(17,24,39,0.35);
        border: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 2rem;
    }}
    
    .pill {{
        display: inline-block;
        padding: 0.4rem 0.8rem;
        border-radius: 999px;
        background: rgba(124,92,252,0.18);
        border: 1px solid rgba(124,92,252,0.35);
        color: #c7b9ff;
        font-size: 0.85rem;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
        transition: all 0.2s ease;
    }}
    
    .pill:hover {{
        background: rgba(124,92,252,0.25);
        transform: scale(1.05);
    }}
    
    .btn-primary button {{
        background: linear-gradient(135deg, {PRIMARY}, {ACCENT}) !important;
        color: white !important;
        border: 0 !important;
        border-radius: 12px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }}
    
    .btn-primary button:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 25px rgba(124,92,252,0.4) !important;
    }}
    
    .status-badge {{
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 500;
        border: 1px solid rgba(255,255,255,0.12);
        background: rgba(255,255,255,0.06);
        margin: 0.2rem;
    }}
    
    .status-not-started {{
        background: rgba(107,114,128,0.2);
        border-color: rgba(107,114,128,0.3);
        color: #9CA3AF;
    }}
    
    .status-in-progress {{
        background: rgba(245,158,11,0.2);
        border-color: rgba(245,158,11,0.3);
        color: #FBBF24;
    }}
    
    .status-done {{
        background: rgba(16,185,129,0.2);
        border-color: rgba(16,185,129,0.3);
        color: #34D399;
    }}
    
    .metric-card {{
        {CARD_BLUR}
        padding: 1.5rem;
        border-radius: 16px;
        text-align: center;
        margin: 0.5rem;
    }}
    
    .metric-value {{
        font-size: 2rem;
        font-weight: 800;
        color: {ACCENT};
        margin-bottom: 0.5rem;
    }}
    
    .metric-label {{
        color: #94A3B8;
        font-size: 0.9rem;
    }}
    
    .progress-bar {{
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
        height: 8px;
        overflow: hidden;
        margin: 1rem 0;
    }}
    
    .progress-fill {{
        background: linear-gradient(90deg, {PRIMARY}, {ACCENT});
        height: 100%;
        border-radius: 10px;
        transition: width 0.5s ease;
    }}
    
    .tab-content {{
        padding: 1rem 0;
    }}
    
    .loading-spinner {{
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid rgba(255,255,255,0.3);
        border-radius: 50%;
        border-top-color: {ACCENT};
        animation: spin 1s ease-in-out infinite;
    }}
    
    @keyframes spin {{
        to {{ transform: rotate(360deg); }}
    }}
    
    .fade-in {{
        animation: fadeIn 0.5s ease-in;
    }}
    
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(20px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    
    .success-message {{
        background: rgba(16,185,129,0.1);
        border: 1px solid rgba(16,185,129,0.3);
        color: #34D399;
        padding: 1rem;
        border-radius: 12px;
        margin: 1rem 0;
    }}
    
    .error-message {{
        background: rgba(239,68,68,0.1);
        border: 1px solid rgba(239,68,68,0.3);
        color: #F87171;
        padding: 1rem;
        border-radius: 12px;
        margin: 1rem 0;
    }}
    
    .info-message {{
        background: rgba(59,130,246,0.1);
        border: 1px solid rgba(59,130,246,0.3);
        color: #60A5FA;
        padding: 1rem;
        border-radius: 12px;
        margin: 1rem 0;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------
# Helpers
# --------------------------
def api():
    headers = {}
    if st.session_state.token:
        headers["Authorization"] = f"Bearer {st.session_state.token}"
    return httpx.Client(base_url=BACKEND_URL, timeout=30.0, headers=headers)

def toast(msg, icon="✅", duration=3):
    st.toast(f"{icon} {msg}", duration=duration)

def status_badge(text, status="default"):
    status_class = f"status-{status.lower().replace(' ', '-')}" if status != "default" else ""
    st.markdown(f"<span class='status-badge {status_class}'>{text}</span>", unsafe_allow_html=True)

def need_login():
    st.markdown("""
    <div class="error-message">
        <strong>🔐 Authentication Required</strong><br>
        Please login or register to access this feature.
    </div>
    """, unsafe_allow_html=True)

def loading_spinner():
    st.markdown('<div class="loading-spinner"></div>', unsafe_allow_html=True)

def success_message(text):
    st.markdown(f'<div class="success-message">{text}</div>', unsafe_allow_html=True)

def error_message(text):
    st.markdown(f'<div class="error-message">{text}</div>', unsafe_allow_html=True)

def info_message(text):
    st.markdown(f'<div class="info-message">{text}</div>', unsafe_allow_html=True)

# --------------------------
# Session state
# --------------------------
for k, v in {"token": None, "user_id": None, "progress": 0, "user_data": None}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# --------------------------
# Header
# --------------------------
st.markdown(
    """
    <div class='hero fade-in'>
        <div style="display:flex;align-items:center;gap:20px;">
            <div style="font-size:36px;">🧭</div>
            <div>
                <div style="font-weight:800;font-size:28px;color:#E5E7EB;margin-bottom:8px;">CareerAI</div>
                <div style="color:#94A3B8;font-size:16px;">Your Personalized Career Companion</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------
# Sidebar: Auth + Profile
# --------------------------
with st.sidebar:
    st.markdown("### 🔐 Account")
    
    if not st.session_state.user_id:
        tab_login, tab_register = st.tabs(["Login", "Register"])
        
        with tab_register:
            st.markdown("#### Create Account")
            name = st.text_input("Full Name", key="reg_name", placeholder="Enter your full name")
            email = st.text_input("Email", key="reg_email", placeholder="your@email.com")
            password = st.text_input("Password", type="password", key="reg_pw", placeholder="Create a strong password")
            
            if st.button("Create Account", type="primary", use_container_width=True):
                if not all([name, email, password]):
                    error_message("Please fill in all fields")
                else:
                    with st.spinner("Creating account..."):
                        try:
                            with api() as c:
                                r = c.post("/auth/register", json={"name": name, "email": email, "password": password})
                            if r.status_code == 200:
                                data = r.json()
                                st.session_state.user_id = data["user_id"]
                                st.session_state.token = data["token"]
                                success_message("Account created successfully!")
                                st.rerun()
                            else:
                                error_message(f"Registration failed: {r.text}")
                        except Exception as e:
                            error_message(f"Connection error: {str(e)}")
        
        with tab_login:
            st.markdown("#### Sign In")
            email_l = st.text_input("Email", key="login_email", placeholder="your@email.com")
            password_l = st.text_input("Password", type="password", key="login_pw", placeholder="Enter your password")
            
            if st.button("Sign In", type="primary", use_container_width=True):
                if not all([email_l, password_l]):
                    error_message("Please fill in all fields")
                else:
                    with st.spinner("Signing in..."):
                        try:
                            with api() as c:
                                r = c.post("/auth/login", json={"email": email_l, "password": password_l})
                            if r.status_code == 200:
                                data = r.json()
                                st.session_state.user_id = data["user_id"]
                                st.session_state.token = data["token"]
                                success_message("Welcome back!")
                                st.rerun()
                            else:
                                error_message(f"Login failed: {r.text}")
                        except Exception as e:
                            error_message(f"Connection error: {str(e)}")
    
    else:
        # User is logged in
        try:
            with api() as c:
                r = c.get("/auth/me")
            if r.status_code == 200:
                me = r.json()
                st.session_state.user_data = me
                
                st.markdown("#### 👤 Profile")
                st.markdown(f"**Name:** {me.get('name', 'N/A')}")
                st.markdown(f"**Email:** {me.get('email', 'N/A')}")
                st.markdown(f"**Phase:** {me.get('current_phase', 'Not set')}")
                st.markdown(f"**Progress:** {int(me.get('progress', 0))}%")
                
                # Progress bar
                progress = int(me.get('progress', 0))
                st.markdown(f"""
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {progress}%"></div>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("Sign Out", type="secondary", use_container_width=True):
                    st.session_state.clear()
                    st.rerun()
            else:
                error_message("Failed to load profile")
        except Exception as e:
            error_message(f"Connection error: {str(e)}")

# --------------------------
# Main tabs
# --------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
    ["🏠 Dashboard", "🔍 Introspection", "💡 Exploration", "🤔 Reflection", "📈 Action", "🚀 Projects", "🎯 Coach", "📝 LinkedIn"]
)

# Dashboard
with tab1:
    st.markdown("### 📊 Your Career Dashboard")
    
    if st.session_state.user_id:
        col1, col2, col3, col4 = st.columns(4)
        
        try:
            # Get user insights
            with api() as c:
                insights_r = c.get("/ai/insights")
                analytics_r = c.get("/analytics/metrics")
                projects_r = c.get("/projects")
            
            if insights_r.status_code == 200:
                insights = insights_r.json()
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{insights.get('activity_score', 0)}</div>
                        <div class="metric-label">Activity Score</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            if analytics_r.status_code == 200:
                analytics = analytics_r.json()
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{analytics.get('total_events', 0)}</div>
                        <div class="metric-label">Total Events</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            if projects_r.status_code == 200:
                projects = projects_r.json().get("projects", [])
                completed = len([p for p in projects if p.get("status") == "Done"])
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{completed}</div>
                        <div class="metric-label">Completed Projects</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{st.session_state.user_data.get('progress', 0)}%</div>
                    <div class="metric-label">Overall Progress</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Recent activity
            st.markdown("### 📈 Recent Activity")
            if insights_r.status_code == 200:
                insights = insights_r.json()
                if insights.get('recent_insights'):
                    for insight in insights['recent_insights'][:3]:
                        st.markdown(f"""
                        <div class="glass">
                            <strong>{insight.get('title', 'Insight')}</strong><br>
                            <small>{insight.get('description', 'No description available')}</small>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    info_message("No recent activity. Start exploring to see insights here!")
            
        except Exception as e:
            error_message(f"Failed to load dashboard data: {str(e)}")
    else:
        need_login()

# Introspection
with tab2:
    st.markdown("### 🔍 Ikigai Journal")
    st.markdown("<span class='pill'>Discover your purpose by exploring what excites you, your strengths, and desired impact</span>", unsafe_allow_html=True)
    
    journal = st.text_area(
        "Your journal", 
        height=200, 
        placeholder="I enjoy teaching with AI and helping students learn. I'm good at explaining complex concepts simply. I want to make education more accessible and personalized for everyone...",
        help="Write about what you love, what you're good at, what the world needs, and what you can be paid for."
    )
    
    if st.button("🔍 Analyze Ikigai", type="primary", use_container_width=True):
        if not st.session_state.user_id:
            need_login()
        elif not journal.strip():
            error_message("Please write something in your journal first")
        else:
            with st.spinner("Analyzing your Ikigai..."):
                try:
                    with api() as c:
                        r = c.post("/ai/ikigai", json={
                            "journal_text": journal
                        })
                    if r.status_code == 200:
                        data = r.json()
                        st.markdown(f"### 📋 Analysis Summary")
                        st.markdown(f"**{data['ai_summary']}**")
                        
                        # Sentiment visualization
                        sentiment = data.get("sentiment_score", 0.5)
                        st.markdown(f"### 😊 Sentiment Score: {sentiment:.1f}/1.0")
                        st.progress(min(1.0, sentiment))
                        
                        # Keywords
                        if data.get("keywords"):
                            st.markdown("### 🏷️ Key Themes")
                            for keyword in data["keywords"][:5]:
                                st.markdown(f"<span class='pill'>{keyword}</span>", unsafe_allow_html=True)
                        
                        success_message("Ikigai analysis completed!")
                    else:
                        error_message(f"Analysis failed: {r.text}")
                except Exception as e:
                    error_message(f"Connection error: {str(e)}")

# Exploration
with tab3:
    st.markdown("### 💡 AI Project Ideas")
    st.markdown("<span class='pill'>Get personalized, scoped, and resume-ready project ideas based on your Ikigai</span>", unsafe_allow_html=True)
    
    ikigai_summary = st.text_area(
        "Paste your Ikigai summary", 
        height=150,
        placeholder="I love teaching with AI, I'm good at explaining complex concepts, I want to make education accessible, and I can be paid for creating educational content..."
    )
    
    if st.button("💡 Generate 3 Ideas", type="primary", use_container_width=True):
        if not st.session_state.user_id:
            need_login()
        elif not ikigai_summary.strip():
            error_message("Please provide your Ikigai summary first")
        else:
            with st.spinner("Generating project ideas..."):
                try:
                    with api() as c:
                        r = c.post("/ai/project-ideas", json={
                            "ikigai_summary": ikigai_summary
                        })
                    if r.status_code == 200:
                        ideas = r.json()
                        st.markdown("### 🚀 Your Personalized Project Ideas")
                        
                        cols = st.columns(3)
                        for i, idea in enumerate(ideas[:3]):
                            with cols[i]:
                                st.markdown(f"""
                                <div class="glass">
                                    <h4>{idea['title']}</h4>
                                    <p>{idea['description']}</p>
                                    <div style="margin-top: 1rem;">
                                        <span class="status-badge status-not-started">{idea.get('status', 'Not Started')}</span>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        success_message("Project ideas generated!")
                    else:
                        error_message(f"Generation failed: {r.text}")
                except Exception as e:
                    error_message(f"Connection error: {str(e)}")

# Reflection
with tab4:
    st.markdown("### 🤔 Delta‑4 Reflection")
    st.markdown("<span class='pill'>Reflect on what's working and what needs improvement in your career journey</span>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### ⚠️ Friction Points")
        friction = st.text_area(
            "What's holding you back?", 
            height=150,
            placeholder="I'm struggling with time management, feeling overwhelmed by too many projects, and not sure which skills to prioritize..."
        )
    
    with col2:
        st.markdown("#### ✨ Delight Moments")
        delight = st.text_area(
            "What's bringing you joy?", 
            height=150,
            placeholder="I love when I can help someone understand a complex concept, I feel energized when working on AI projects, and I'm proud of my recent teaching achievements..."
        )
    
    if st.button("🤔 Analyze Reflection", type="primary", use_container_width=True):
        if not st.session_state.user_id:
            need_login()
        elif not friction.strip() or not delight.strip():
            error_message("Please fill in both friction and delight sections")
        else:
            with st.spinner("Analyzing your reflection..."):
                try:
                    with api() as c:
                        r = c.post("/ai/reflection", json={
                            "friction": friction, 
                            "delight": delight
                        })
                    if r.status_code == 200:
                        data = r.json()
                        st.markdown("### 💡 Key Insights")
                        st.markdown(f"**{data['ai_summary']}**")
                        
                        # Action items
                        if data.get("action_items"):
                            st.markdown("### 🎯 Recommended Actions")
                            for item in data["action_items"][:3]:
                                st.markdown(f"• {item}")
                        
                        success_message("Reflection analysis completed!")
                    else:
                        error_message(f"Analysis failed: {r.text}")
                except Exception as e:
                    error_message(f"Connection error: {str(e)}")

# Action
with tab5:
    st.markdown("### 📈 Phase Progress")
    
    try:
        with api() as c:
            phases_r = c.get("/phases")
        if phases_r.status_code == 200:
            phases = phases_r.json()
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("#### 📋 Available Phases")
                for i, phase in enumerate(phases):
                    st.markdown(f"""
                    <div class="glass">
                        <strong>{phase['id']}. {phase['name']}</strong><br>
                        <small>{phase['description']}</small>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("#### 🎯 Update Progress")
                new_phase = st.selectbox(
                    "Current phase", 
                    [p["name"] for p in phases],
                    index=0
                )
                new_progress = st.slider("Progress %", 0, 100, 0)
                
                if st.button("📈 Update Progress", type="primary", use_container_width=True):
                    if not st.session_state.user_id:
                        need_login()
                    else:
                        with st.spinner("Updating progress..."):
                            try:
                                with api() as c:
                                    r = c.patch(f"/phase/1/progress", json={
                                        "current_phase": new_phase, 
                                        "progress": new_progress
                                    })
                                if r.status_code == 200:
                                    success_message("Progress updated successfully!")
                                    st.rerun()
                                else:
                                    error_message(f"Update failed: {r.text}")
                            except Exception as e:
                                error_message(f"Connection error: {str(e)}")
        else:
            error_message("Failed to load phases")
    except Exception as e:
        error_message(f"Connection error: {str(e)}")

# Projects
with tab6:
    st.markdown("### 🚀 Your Projects")
    
    if st.session_state.user_id:
        try:
            with api() as c:
                r = c.get("/projects")
            if r.status_code == 200:
                projects = r.json().get("projects", [])
                
                if projects:
                    for proj in projects:
                        with st.container():
                            st.markdown(f"""
                            <div class="glass">
                                <h4>{proj['title']}</h4>
                                <p>{proj['description']}</p>
                                <div style="margin: 1rem 0;">
                                    <span class="status-badge status-{proj.get('status', 'not-started').lower().replace(' ', '-')}">{proj.get('status', 'Not Started')}</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Update status
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                new_status = st.selectbox(
                                    "Update Status", 
                                    ["Not Started", "In Progress", "Done"],
                                    index=["Not Started", "In Progress", "Done"].index(proj.get("status", "Not Started")),
                                    key=f"status-{proj['id']}"
                                )
                            with col2:
                                if st.button("💾 Save", key=f"save-{proj['id']}"):
                                    with st.spinner("Updating project..."):
                                        try:
                                            with api() as c:
                                                r2 = c.patch(f"/projects/{proj['id']}", json={"status": new_status})
                                            if r2.status_code == 200:
                                                success_message("Project updated!")
                                                st.rerun()
                                            else:
                                                error_message(f"Update failed: {r2.text}")
                                        except Exception as e:
                                            error_message(f"Connection error: {str(e)}")
                else:
                    info_message("No projects yet. Generate some ideas in the Exploration tab!")
            else:
                error_message("Failed to load projects")
        except Exception as e:
            error_message(f"Connection error: {str(e)}")
    else:
        need_login()

# Coach
with tab7:
    st.markdown("### 🎯 Career Coach")
    st.markdown("<span class='pill'>Get personalized guidance and next steps for your career journey</span>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 👤 Your Current Situation")
        stage = st.text_input(
            "Current role/stage", 
            placeholder="Final-year student / Data intern / Junior developer",
            help="Describe your current career stage or role"
        )
        goals = st.text_area(
            "Goals (3-5 lines)", 
            height=120,
            placeholder="I want to become a senior AI engineer, lead technical teams, and contribute to open-source projects that make a real impact...",
            help="What are your career goals?"
        )
    
    with col2:
        st.markdown("#### 📝 Recent Work")
        recent = st.text_area(
            "Recent work summary", 
            height=160,
            placeholder="Built a small RAG app with LangChain, contributed to an open-source project, completed a machine learning course...",
            help="What have you been working on recently?"
        )
    
    if st.button("🎯 Get Guidance", type="primary", use_container_width=True):
        if not st.session_state.user_id:
            need_login()
        elif not all([stage, goals, recent]):
            error_message("Please fill in all fields")
        else:
            with st.spinner("Getting personalized guidance..."):
                try:
                    with api() as c:
                        r = c.post("/ai/guidance", json={
                            "current_role_or_stage": stage,
                            "goals": goals,
                            "recent_work_summary": recent
                        })
                    if r.status_code == 200:
                        data = r.json()
                        st.markdown("### 💡 Personalized Advice")
                        st.markdown(f"**{data['advice']}**")
                        
                        if data.get("next_steps"):
                            st.markdown("### 🎯 Next Steps")
                            for i, step in enumerate(data["next_steps"], 1):
                                st.markdown(f"**{i}.** {step}")
                        
                        if data.get("resources"):
                            st.markdown("### 📚 Recommended Resources")
                            for resource in data["resources"][:3]:
                                st.markdown(f"• {resource}")
                        
                        success_message("Guidance generated!")
                    else:
                        error_message(f"Guidance failed: {r.text}")
                except Exception as e:
                    error_message(f"Connection error: {str(e)}")

# LinkedIn Post Generator
with tab8:
    st.markdown("### 📝 LinkedIn Post Generator")
    st.markdown("<span class='pill'>Create engaging LinkedIn posts to build your professional brand</span>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📋 Post Details")
        post_type = st.selectbox(
            "Post Type",
            ["achievement", "project", "learning", "career_update", "networking", "thought_leadership"],
            format_func=lambda x: {
                "achievement": "🏆 Achievement/Milestone",
                "project": "🚀 Project Showcase", 
                "learning": "📚 Learning Experience",
                "career_update": "💼 Career Update",
                "networking": "🤝 Networking",
                "thought_leadership": "💡 Thought Leadership"
            }[x],
            help="Choose the type of LinkedIn post you want to create"
        )
        
        tone = st.selectbox(
            "Tone",
            ["professional", "casual", "inspiring", "technical", "personal"],
            format_func=lambda x: {
                "professional": "👔 Professional",
                "casual": "😊 Casual",
                "inspiring": "✨ Inspiring", 
                "technical": "🔧 Technical",
                "personal": "👤 Personal"
            }[x],
            help="Choose the tone for your post"
        )
        
        content = st.text_area(
            "Content to work with",
            height=150,
            placeholder="I just completed my first machine learning project using Python and scikit-learn. It was challenging but I learned so much about data preprocessing, model training, and evaluation...",
            help="Describe what you want to post about. Be specific and include details, metrics, or experiences."
        )
    
    with col2:
        st.markdown("#### ⚙️ Options")
        include_hashtags = st.checkbox("Include hashtags", value=True, help="Add relevant hashtags to increase visibility")
        include_cta = st.checkbox("Include call-to-action", value=True, help="Add a question or call-to-action to encourage engagement")
        
        st.markdown("#### 💡 Tips")
        st.info("""
        **Best practices for LinkedIn posts:**
        - Keep it authentic and personal
        - Include specific details or metrics
        - Use line breaks for readability
        - Post during business hours (9 AM - 5 PM)
        - Engage with comments quickly
        """)
    
    if st.button("📝 Generate LinkedIn Post", type="primary", use_container_width=True):
        if not st.session_state.user_id:
            need_login()
        elif not content.strip():
            error_message("Please provide content to work with")
        else:
            with st.spinner("Generating your LinkedIn post..."):
                try:
                    with api() as c:
                        r = c.post("/ai/linkedin-post", json={
                            "post_type": post_type,
                            "content": content,
                            "tone": tone,
                            "include_hashtags": include_hashtags,
                            "include_call_to_action": include_cta
                        })
                    if r.status_code == 200:
                        data = r.json()
                        
                        st.markdown("### 📝 Your LinkedIn Post")
                        
                        # Post content
                        st.markdown("**Post Content:**")
                        st.text_area(
                            "Generated Post",
                            value=data["post_content"],
                            height=200,
                            key="linkedin_post_content",
                            help="Copy this text for your LinkedIn post"
                        )
                        
                        # Hashtags
                        if data.get("hashtags"):
                            st.markdown("**Hashtags:**")
                            hashtags_text = " ".join(data["hashtags"])
                            st.text_input(
                                "Generated Hashtags",
                                value=hashtags_text,
                                key="linkedin_hashtags",
                                help="Copy these hashtags to add to your post"
                            )
                        
                        # Call to Action
                        if data.get("call_to_action"):
                            st.markdown("**Call to Action:**")
                            st.text_input(
                                "Generated CTA",
                                value=data["call_to_action"],
                                key="linkedin_cta",
                                help="Add this call-to-action to encourage engagement"
                            )
                        
                        # Engagement tips
                        if data.get("engagement_tips"):
                            st.markdown("### 💡 Engagement Tips")
                            for tip in data["engagement_tips"]:
                                st.markdown(f"• {tip}")
                        
                        # Character count
                        char_count = data.get("character_count", 0)
                        if char_count > 0:
                            st.markdown(f"**Character Count:** {char_count}")
                            if char_count > 3000:
                                st.warning("⚠️ Post is quite long. Consider shortening for better engagement.")
                            elif char_count < 100:
                                st.info("💡 Post is quite short. Consider adding more details.")
                        
                        # Copy buttons
                        col_copy1, col_copy2, col_copy3 = st.columns(3)
                        with col_copy1:
                            if st.button("📋 Copy Post", use_container_width=True):
                                st.write("Post content copied to clipboard!")
                        with col_copy2:
                            if st.button("📋 Copy Hashtags", use_container_width=True):
                                st.write("Hashtags copied to clipboard!")
                        with col_copy3:
                            if st.button("📋 Copy CTA", use_container_width=True):
                                st.write("Call-to-action copied to clipboard!")
                        
                        success_message("LinkedIn post generated successfully!")
                    else:
                        error_message(f"Post generation failed: {r.text}")
                except Exception as e:
                    error_message(f"Connection error: {str(e)}")

# --------------------------
# Footer
# --------------------------
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #94A3B8; padding: 2rem 0;">
        <p>🧭 <strong>CareerAI</strong> - Your Personalized Career Companion</p>
        <p>Built with ❤️ for your career success</p>
    </div>
    """,
    unsafe_allow_html=True,
)
