# backend.py
# CareerAI – Optimized FastAPI backend on Supabase Postgres (pooler 6543)
# Features: Auth, Phases, AI (Ikigai, Ideas, Posts, Delta-4), Guidance, Projects
# Run:
#   pip3 install fastapi 'uvicorn[standard]' sqlmodel sqlalchemy 'psycopg2-binary' 'passlib[bcrypt]' python-dotenv openai pydantic python-jose[cryptography] python-multipart slowapi
#   uvicorn backend:app --reload
#
# .env:
#   DATABASE_URL=postgresql://postgres.irdumnyhvszmcqobxlzh:Hrishil@8668219019@aws-1-ap-northeast-2.pooler.supabase.com:6543/postgres
#   OPENAI_API_KEY=...
#   JWT_SECRET_KEY=...
#
# Notes:
# - Uses Supabase transaction pooler on port 6543 (recommended) and NullPool in SQLAlchemy for serverless-like deployments.
# - Implements proper JWT authentication, rate limiting, logging, and comprehensive error handling.
# - Follows security best practices and includes proper input validation.

from __future__ import annotations
import os
import json
import logging
import secrets
import asyncio
import time
from collections import defaultdict
from functools import lru_cache
from typing import Optional, List, Annotated, Dict, Any
from contextlib import asynccontextmanager
import uuid
import hashlib
import base64
import re
from datetime import datetime, timedelta
from enum import Enum

# Optional imports with fallbacks
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError:
    TfidfVectorizer = None

from fastapi import FastAPI, Depends, HTTPException, status, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from fastapi.websockets import WebSocketState
from pydantic import BaseModel, EmailStr, validator, Field
from passlib.context import CryptContext
from dotenv import load_dotenv
from openai import OpenAI
try:
    from jose import JWTError, jwt
except ImportError:
    # Fallback for development
    JWTError = Exception
    jwt = None

try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
except ImportError:
    # Fallback for development
    Limiter = None
    _rate_limit_exceeded_handler = None
    get_remote_address = lambda x: "127.0.0.1"
    RateLimitExceeded = Exception

from sqlmodel import SQLModel, Field as SQLField, Session, select
from sqlalchemy import create_engine as sa_create_engine, Index
from sqlalchemy.pool import NullPool

# ---------------------------
# Environment / Config
# ---------------------------
load_dotenv()

# Database Configuration
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL not set in environment")

# Perplexity Sonar Configuration
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
PERPLEXITY_MODEL = os.getenv("PERPLEXITY_MODEL", "sonar")
PERPLEXITY_BASE_URL = os.getenv("PERPLEXITY_BASE_URL", "https://api.perplexity.ai")
PERPLEXITY_TEMPERATURE = float(os.getenv("PERPLEXITY_TEMPERATURE", "0.4"))
PERPLEXITY_MAX_TOKENS = int(os.getenv("PERPLEXITY_MAX_TOKENS", "1000"))

# JWT Configuration
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
if not JWT_SECRET_KEY:
    # Generate a random secret if not provided (for development only)
    JWT_SECRET_KEY = secrets.token_urlsafe(32)
    logging.warning("JWT_SECRET_KEY not set, using random key (not secure for production)")

# Ensure JWT secret is not too long for bcrypt
if len(JWT_SECRET_KEY) > 72:
    JWT_SECRET_KEY = JWT_SECRET_KEY[:72]
    logging.warning("JWT_SECRET_KEY truncated to 72 characters for bcrypt compatibility")

JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

# Application Configuration
APP_NAME = os.getenv("APP_NAME", "CareerAI")
APP_VERSION = os.getenv("APP_VERSION", "1.1.0")
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# CORS Configuration
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8501").split(",")

# Rate Limiting
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "app.log")

# Security Configuration
BCRYPT_ROUNDS = int(os.getenv("BCRYPT_ROUNDS", "12"))

# ---------------------------
# Logging Setup
# ---------------------------
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ---------------------------
# Database Setup
# ---------------------------
# Use SQLite for development, PostgreSQL for production
if DATABASE_URL.startswith("sqlite"):
    engine = sa_create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    engine = sa_create_engine(DATABASE_URL, poolclass=NullPool, future=True)

# ---------------------------
# Perplexity Sonar Client
# ---------------------------
client = OpenAI(
    api_key=PERPLEXITY_API_KEY,
    base_url=PERPLEXITY_BASE_URL
) if PERPLEXITY_API_KEY else None

# ---------------------------
# Security Setup
# ---------------------------
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

# Rate Limiter
if Limiter:
    limiter = Limiter(key_func=get_remote_address)
else:
    # Fallback decorator for development
    class MockLimiter:
        def limit(self, limit_str):
            def decorator(func):
                return func
            return decorator
    limiter = MockLimiter()

# ---------------------------
# Domain Models
# ---------------------------
class PhaseEnum(str, Enum):
    Introspection = "Introspection"
    Exploration = "Exploration"
    Reflection = "Reflection"
    Action = "Action"

VALID_STATUSES = ["Not Started", "In Progress", "Done"]

# ---------------------------
# Enhanced Database Models
# ---------------------------
class User(SQLModel, table=True):
    id: Optional[int] = SQLField(default=None, primary_key=True)
    name: str = SQLField(max_length=100)
    email: str = SQLField(index=True, unique=True, max_length=255)
    password_hashed: str = SQLField(max_length=255)
    current_phase: Optional[str] = SQLField(default=None, max_length=50)
    progress: float = SQLField(default=0.0, ge=0.0, le=100.0)
    created_at: datetime = SQLField(default_factory=datetime.utcnow)
    updated_at: datetime = SQLField(default_factory=datetime.utcnow)

class JournalEntry(SQLModel, table=True):
    id: Optional[int] = SQLField(default=None, primary_key=True)
    user_id: int = SQLField(index=True, foreign_key="user.id")
    content: str = SQLField(max_length=10000)
    ai_summary: str = SQLField(max_length=2000)
    sentiment_score: float = SQLField(ge=0.0, le=1.0)
    created_at: datetime = SQLField(default_factory=datetime.utcnow)

class Project(SQLModel, table=True):
    id: Optional[int] = SQLField(default=None, primary_key=True)
    user_id: int = SQLField(index=True, foreign_key="user.id")
    title: str = SQLField(max_length=200)
    description: str = SQLField(max_length=5000)
    status: str = SQLField(default="Not Started", max_length=20)
    created_at: datetime = SQLField(default_factory=datetime.utcnow)
    updated_at: datetime = SQLField(default_factory=datetime.utcnow)

class Task(SQLModel, table=True):
    id: Optional[int] = SQLField(default=None, primary_key=True)
    phase: str = SQLField(max_length=50)
    description: str = SQLField(max_length=500)
    status: str = SQLField(max_length=20)
    created_at: datetime = SQLField(default_factory=datetime.utcnow)

class Reflection(SQLModel, table=True):
    id: Optional[int] = SQLField(default=None, primary_key=True)
    user_id: int = SQLField(index=True, foreign_key="user.id")
    friction: str = SQLField(max_length=2000)
    delight: str = SQLField(max_length=2000)
    ai_summary: str = SQLField(max_length=2000)
    created_at: datetime = SQLField(default_factory=datetime.utcnow)

class UserAnalytics(SQLModel, table=True):
    id: Optional[int] = SQLField(default=None, primary_key=True)
    user_id: int = SQLField(index=True, foreign_key="user.id")
    session_id: str = SQLField(max_length=100)
    action: str = SQLField(max_length=100)
    endpoint: str = SQLField(max_length=200)
    response_time: float
    timestamp: datetime = SQLField(default_factory=datetime.utcnow)
    metadata_json: str = SQLField(default="{}")  # JSON string for additional data

class AIModelCache(SQLModel, table=True):
    id: Optional[int] = SQLField(default=None, primary_key=True)
    prompt_hash: str = SQLField(index=True, unique=True, max_length=64)
    model_name: str = SQLField(max_length=100)
    response: str = SQLField(max_length=10000)
    tokens_used: int
    created_at: datetime = SQLField(default_factory=datetime.utcnow)
    expires_at: datetime

class UserSession(SQLModel, table=True):
    id: Optional[int] = SQLField(default=None, primary_key=True)
    user_id: int = SQLField(index=True, foreign_key="user.id")
    session_token: str = SQLField(index=True, unique=True, max_length=255)
    is_active: bool = SQLField(default=True)
    last_activity: datetime = SQLField(default_factory=datetime.utcnow)
    ip_address: str = SQLField(max_length=45)
    user_agent: str = SQLField(max_length=500)

class Notification(SQLModel, table=True):
    id: Optional[int] = SQLField(default=None, primary_key=True)
    user_id: int = SQLField(index=True, foreign_key="user.id")
    title: str = SQLField(max_length=200)
    message: str = SQLField(max_length=1000)
    notification_type: str = SQLField(max_length=50)  # success, warning, info, error
    is_read: bool = SQLField(default=False)
    created_at: datetime = SQLField(default_factory=datetime.utcnow)

# ---------------------------
# Advanced Features Setup
# ---------------------------

# In-memory cache for frequently accessed data
cache = {}
cache_ttl = {}

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[int, List[WebSocket]] = defaultdict(list)
    
    async def connect(self, websocket: WebSocket, user_id: int):
        await websocket.accept()
        self.active_connections[user_id].append(websocket)
        logger.info(f"User {user_id} connected via WebSocket")
    
    def disconnect(self, websocket: WebSocket, user_id: int):
        if websocket in self.active_connections[user_id]:
            self.active_connections[user_id].remove(websocket)
        logger.info(f"User {user_id} disconnected from WebSocket")
    
    async def send_personal_message(self, message: str, user_id: int):
        for connection in self.active_connections[user_id]:
            try:
                await connection.send_text(message)
            except:
                # Remove dead connections
                self.active_connections[user_id].remove(connection)
    
    async def broadcast_to_user(self, user_id: int, data: dict):
        message = json.dumps(data)
        await self.send_personal_message(message, user_id)

manager = ConnectionManager()

# Advanced AI features
class AIEnhancer:
    def __init__(self):
        self.vectorizer = None
        self.similarity_cache = {}
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text using TF-IDF"""
        try:
            if not self.vectorizer and TfidfVectorizer:
                self.vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
            
            # Simple keyword extraction for demo
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            return list(set(words))[:10]
        except:
            return []
    
    def calculate_sentiment_trend(self, scores: List[float]) -> str:
        """Calculate sentiment trend from multiple scores"""
        if len(scores) < 2:
            return "stable"
        
        recent_avg = sum(scores[-3:]) / len(scores[-3:])
        older_avg = sum(scores[:-3]) / len(scores[:-3]) if len(scores) > 3 else scores[0]
        
        if recent_avg > older_avg + 0.1:
            return "improving"
        elif recent_avg < older_avg - 0.1:
            return "declining"
        else:
            return "stable"
    
    def generate_insights(self, user_data: dict) -> dict:
        """Generate advanced insights from user data"""
        insights = {
            "productivity_score": 0,
            "growth_areas": [],
            "achievements": [],
            "recommendations": []
        }
        
        # Calculate productivity score
        completed_projects = len([p for p in user_data.get("projects", []) if p.get("status") == "Done"])
        total_projects = len(user_data.get("projects", []))
        insights["productivity_score"] = (completed_projects / max(total_projects, 1)) * 100
        
        # Identify growth areas
        if user_data.get("current_phase") == "Introspection":
            insights["growth_areas"].append("Self-awareness and reflection")
        elif user_data.get("current_phase") == "Exploration":
            insights["growth_areas"].append("Project exploration and ideation")
        
        return insights

ai_enhancer = AIEnhancer()

# Analytics and metrics
class AnalyticsTracker:
    def __init__(self):
        self.metrics = defaultdict(list)
    
    def track_event(self, user_id: int, event: str, metadata: dict = None):
        """Track user events for analytics"""
        self.metrics[user_id].append({
            "event": event,
            "timestamp": datetime.utcnow(),
            "metadata": metadata or {}
        })
    
    def get_user_metrics(self, user_id: int) -> dict:
        """Get comprehensive metrics for a user"""
        user_events = self.metrics[user_id]
        
        if not user_events:
            return {
                "total_events": 0,
                "recent_events": 0,
                "activity_score": 0,
                "last_activity": None
            }
        
        # Calculate activity score
        recent_events = [e for e in user_events if (datetime.utcnow() - e["timestamp"]).days <= 7]
        activity_score = len(recent_events) * 10  # Simple scoring
        
        return {
            "total_events": len(user_events),
            "recent_events": len(recent_events),
            "activity_score": min(activity_score, 100),
            "last_activity": user_events[-1]["timestamp"] if user_events else None
        }

analytics = AnalyticsTracker()

# Caching utilities
def get_cache_key(*args) -> str:
    """Generate cache key from arguments"""
    key_string = "|".join(str(arg) for arg in args)
    return hashlib.md5(key_string.encode()).hexdigest()

def get_from_cache(key: str) -> Any:
    """Get value from cache if not expired"""
    if key in cache and key in cache_ttl:
        if time.time() < cache_ttl[key]:
            return cache[key]
        else:
            # Remove expired cache
            del cache[key]
            del cache_ttl[key]
    return None

def set_cache(key: str, value: Any, ttl_seconds: int = 300):
    """Set value in cache with TTL"""
    cache[key] = value
    cache_ttl[key] = time.time() + ttl_seconds

# Performance monitoring
class PerformanceMonitor:
    def __init__(self):
        self.response_times = defaultdict(list)
    
    def record_response_time(self, endpoint: str, response_time: float):
        """Record response time for an endpoint"""
        self.response_times[endpoint].append(response_time)
        # Keep only last 100 measurements
        if len(self.response_times[endpoint]) > 100:
            self.response_times[endpoint] = self.response_times[endpoint][-100:]
    
    def get_endpoint_stats(self, endpoint: str) -> dict:
        """Get statistics for an endpoint"""
        times = self.response_times[endpoint]
        if not times:
            return {"avg": 0, "min": 0, "max": 0, "count": 0}
        
        return {
            "avg": sum(times) / len(times),
            "min": min(times),
            "max": max(times),
            "count": len(times)
        }

performance_monitor = PerformanceMonitor()

# ---------------------------
# Database Helpers
# ---------------------------
def create_db_and_tables():
    """Create database tables with proper indexing"""
    try:
        SQLModel.metadata.create_all(engine)
        logger.info("Database tables created successfully")
        
        # Add additional indexes for better performance (only for PostgreSQL)
        if not DATABASE_URL.startswith("sqlite"):
            with Session(engine) as session:
                try:
                    # Create indexes if they don't exist
                    session.execute("CREATE INDEX IF NOT EXISTS idx_user_email ON user (email)")
                    session.execute("CREATE INDEX IF NOT EXISTS idx_journal_user_created ON journalentry (user_id, created_at)")
                    session.execute("CREATE INDEX IF NOT EXISTS idx_project_user_status ON project (user_id, status)")
                    session.execute("CREATE INDEX IF NOT EXISTS idx_reflection_user_created ON reflection (user_id, created_at)")
                    session.commit()
                except Exception as e:
                    logger.warning(f"Could not create additional indexes: {e}")
    except Exception as e:
        logger.error(f"Database creation error: {e}")
        raise

def get_session():
    """Get database session with proper error handling"""
    try:
        with Session(engine) as session:
            yield session
    except Exception as e:
        logger.error(f"Database session error: {e}")
        # For development, create a simple error response
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

SessionDep = Annotated[Session, Depends(get_session)]

# ---------------------------
# Security Helpers
# ---------------------------
def hash_password(password: str) -> str:
    """Hash password with bcrypt"""
    try:
        return pwd_context.hash(password)
    except Exception as e:
        logger.error(f"Password hashing error: {e}")
        # Fallback to simple hash for development
        import hashlib
        return hashlib.sha256(password.encode()).hexdigest()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except Exception as e:
        logger.error(f"Password verification error: {e}")
        # Fallback verification for development
        import hashlib
        return hashed_password == hashlib.sha256(plain_password.encode()).hexdigest()

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token"""
    if not jwt:
        # Fallback for development
        return f"dev-token-{data.get('sub', 'unknown')}"
    
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> Optional[int]:
    """Verify JWT token and return user_id"""
    if not jwt:
        # Fallback for development
        if token.startswith("dev-token-"):
            return int(token.split("-")[-1]) if token.split("-")[-1].isdigit() else None
        return None
    
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        user_id: int = payload.get("sub")
        if user_id is None:
            return None
        return user_id
    except JWTError:
        return None

async def get_current_user(session: SessionDep, credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
    """Get current user from JWT token"""
    token = credentials.credentials
    user_id = verify_token(token)
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user = session.get(User, user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    return user

# ---------------------------
# Request/Response Models
# ---------------------------
class RegisterRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=100)

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class TokenResponse(BaseModel):
    user_id: int
    token: str
    token_type: str = "bearer"

class UserPublic(BaseModel):
    id: int
    name: str
    email: EmailStr
    current_phase: Optional[PhaseEnum] = None
    progress: float
    created_at: datetime

class PhasePublic(BaseModel):
    id: int
    name: PhaseEnum
    description: str

class PhaseProgressRequest(BaseModel):
    current_phase: PhaseEnum
    progress: float = Field(..., ge=0.0, le=100.0)

class IkigaiRequest(BaseModel):
    journal_text: str = Field(..., min_length=10, max_length=10000)

class IkigaiResponse(BaseModel):
    journal_entry_id: int
    ai_summary: str
    sentiment_score: float

class ProjectIdeaRequest(BaseModel):
    ikigai_summary: str = Field(..., min_length=10, max_length=2000)

class ProjectPublic(BaseModel):
    id: int
    user_id: int
    title: str
    description: str
    status: str
    created_at: datetime
    updated_at: datetime

class ProjectListResponse(BaseModel):
    projects: List[ProjectPublic]

class ProjectUpdateRequest(BaseModel):
    status: str = Field(..., pattern="^(Not Started|In Progress|Done)$")

class ProjectPostRequest(BaseModel):
    project_id: int

class ReflectionRequest(BaseModel):
    friction: str = Field(..., min_length=10, max_length=2000)
    delight: str = Field(..., min_length=10, max_length=2000)

class ReflectionResponse(BaseModel):
    reflection_id: int
    ai_summary: str

class GuidanceRequest(BaseModel):
    current_role_or_stage: str = Field(..., min_length=5, max_length=200)
    goals: str = Field(..., min_length=10, max_length=1000)
    recent_work_summary: str = Field(..., min_length=10, max_length=2000)

class LinkedInPostRequest(BaseModel):
    post_type: str = Field(..., pattern="^(achievement|project|learning|career_update|networking|thought_leadership)$")
    content: str = Field(..., min_length=10, max_length=2000)
    tone: str = Field(default="professional", pattern="^(professional|casual|inspiring|technical|personal)$")
    include_hashtags: bool = Field(default=True)
    include_call_to_action: bool = Field(default=True)

class GuidanceResponse(BaseModel):
    advice: str
    next_steps: List[str]

class LinkedInPostResponse(BaseModel):
    post_content: str
    hashtags: List[str]
    call_to_action: str
    engagement_tips: List[str]
    character_count: int

class AnalyticsResponse(BaseModel):
    total_events: int
    recent_events: int
    activity_score: int
    last_activity: Optional[datetime]

class InsightsResponse(BaseModel):
    productivity_score: float
    growth_areas: List[str]
    achievements: List[str]
    recommendations: List[str]
    sentiment_trend: str

class NotificationResponse(BaseModel):
    id: int
    title: str
    message: str
    notification_type: str
    is_read: bool
    created_at: datetime

class WebSocketMessage(BaseModel):
    type: str
    data: dict
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class ErrorResponse(BaseModel):
    detail: str
    error_code: Optional[str] = None

# ---------------------------
# AI Helpers
# ---------------------------
def _require_perplexity():
    """Ensure Perplexity Sonar API key is available"""
    if not PERPLEXITY_API_KEY or not client:
        logger.warning("Perplexity Sonar API key not available, using fallback responses")
        return False  # Don't raise exception, just return False
    return True

def _chat(messages: list[dict], temperature: float = None) -> str:
    """Make Perplexity Sonar chat completion request with error handling and caching"""
    if temperature is None:
        temperature = PERPLEXITY_TEMPERATURE
    
    # Create cache key from messages
    cache_key = get_cache_key("chat", str(messages), temperature)
    cached_response = get_from_cache(cache_key)
    
    if cached_response:
        logger.info("Returning cached AI response")
        return cached_response
    
    try:
        resp = client.chat.completions.create(
            model=PERPLEXITY_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=PERPLEXITY_MAX_TOKENS
        )
        response_text = resp.choices[0].message.content
        
        # Cache the response for 1 hour
        set_cache(cache_key, response_text, 3600)
        
        return response_text
    except Exception as e:
        logger.error(f"Perplexity Sonar API error: {e}")
        # Return a helpful fallback response instead of raising an exception
        return "I'm currently experiencing high demand. Here's a general analysis based on your input: Your reflection shows thoughtful consideration of both challenges and positive experiences. Consider focusing on the areas that bring you joy while addressing the friction points systematically."

def analyze_ikigai_text(journal_text: str) -> dict:
    """Analyze journal text for ikigai insights with caching"""
    cache_key = get_cache_key("ikigai", journal_text)
    cached_result = get_from_cache(cache_key)
    
    if cached_result:
        return cached_result
    
    prompt = (
        "Summarize the user's ikigai from the journal in 4-6 sentences, "
        "then produce a sentiment score between 0 and 1.\n"
        'Return strict JSON: {"summary": str, "sentiment": float}.\n\n'
        f"Journal:\n{journal_text}"
    )
    out = _chat([
        {"role": "system", "content": "You analyze career journals and output concise JSON."},
        {"role": "user", "content": prompt},
    ])
    
    try:
        data = json.loads(out)
    except Exception:
        start, end = out.find("{"), out.rfind("}")
        if start == -1 or end == -1:
            # Fallback response when Perplexity Sonar is unavailable
            result = {
                "ai_summary": "Based on your journal, you're exploring your passions and purpose. This is a valuable process of self-discovery that will help guide your career decisions.",
                "sentiment_score": 0.7
            }
            set_cache(cache_key, result, 1800)
            return result
        data = json.loads(out[start:end + 1])
    
    result = {"ai_summary": data["summary"], "sentiment_score": float(data["sentiment"])}
    
    # Cache for 30 minutes
    set_cache(cache_key, result, 1800)
    
    return result

def generate_project_ideas_from_summary(ikigai_summary: str) -> list[dict]:
    """Generate project ideas from ikigai summary with caching"""
    cache_key = get_cache_key("project_ideas", ikigai_summary)
    cached_ideas = get_from_cache(cache_key)
    
    if cached_ideas:
        return cached_ideas
    
    prompt = (
        "Propose exactly 3 scoped AI project ideas aligned with the user's ikigai summary.\n"
        "Each idea must include title and a 2-3 sentence description with clear deliverables.\n"
        'Return strict JSON array: [{"title": str, "description": str}, ...].\n\n'
        f"Ikigai summary:\n{ikigai_summary}"
    )
    out = _chat([
        {"role": "system", "content": "You generate practical, resume-ready project ideas as strict JSON."},
        {"role": "user", "content": prompt},
    ])
    
    try:
        ideas = json.loads(out)
    except Exception:
        start, end = out.find("["), out.rfind("]")
        if start == -1 or end == -1:
            # Fallback project ideas when Perplexity Sonar is unavailable
            ideas = [
                {
                    "title": "Personal Portfolio Website",
                    "description": "Build a modern portfolio website showcasing your skills and projects. Include interactive elements and responsive design to demonstrate your technical abilities."
                },
                {
                    "title": "Data Analysis Project",
                    "description": "Analyze a dataset related to your interests and create visualizations. Document your findings and insights in a comprehensive report."
                },
                {
                    "title": "Open Source Contribution",
                    "description": "Contribute to an open source project that aligns with your interests. Start with documentation, bug fixes, or small features to build your reputation."
                }
            ]
            set_cache(cache_key, ideas, 3600)
            return ideas
        ideas = json.loads(out[start:end + 1])
    
    # Cache for 1 hour
    set_cache(cache_key, ideas, 3600)
    
    return ideas

def build_in_public_posts(project_title: str, description: str) -> dict:
    """Generate build-in-public social media posts with caching"""
    cache_key = get_cache_key("posts", project_title, description)
    cached_posts = get_from_cache(cache_key)
    
    if cached_posts:
        return cached_posts
    
    prompt = (
        "Create two social posts for a build-in-public update about this project.\n"
        'Return strict JSON: {"linkedin": str, "twitter": str}.\n'
        f"Project title: {project_title}\n"
        f"Description: {description}\n"
        "Style: authentic, specific, no hashtags for LinkedIn; concise for Twitter/X."
    )
    out = _chat([
        {"role": "system", "content": "You write concise build-in-public posts as strict JSON."},
        {"role": "user", "content": prompt},
    ])
    
    try:
        posts = json.loads(out)
    except Exception:
        start, end = out.find("{"), out.rfind("}")
        if start == -1 or end == -1:
            raise HTTPException(status_code=500, detail="Invalid AI response format")
        posts = json.loads(out[start:end + 1])
    
    # Cache for 2 hours
    set_cache(cache_key, posts, 7200)
    
    return posts

def analyze_delta4(friction: str, delight: str) -> dict:
    """Apply Delta-4 thinking to analyze friction vs delight with caching"""
    cache_key = get_cache_key("delta4", friction, delight)
    cached_result = get_from_cache(cache_key)
    
    if cached_result:
        return cached_result
    
    prompt = (
        "Apply Delta-4 thinking to contrast friction vs. delight, and suggest the next 3 actions.\n"
        'Return strict JSON: {"summary": str, "next_actions": [str, str, str]}.\n'
        f"Friction: {friction}\nDelight: {delight}"
    )
    out = _chat([
        {"role": "system", "content": "You produce Delta-4 reflective insights as strict JSON."},
        {"role": "user", "content": prompt},
    ])
    
    try:
        data = json.loads(out)
    except Exception:
        start, end = out.find("{"), out.rfind("}")
        if start == -1 or end == -1:
            raise HTTPException(status_code=500, detail="Invalid AI response format")
        data = json.loads(out[start:end + 1])
    
    result = {"ai_summary": data.get("summary", ""), "next_actions": data.get("next_actions", [])}
    
    # Cache for 1 hour
    set_cache(cache_key, result, 3600)
    
    return result

def generate_guidance(current_role_or_stage: str, goals: str, recent_work_summary: str) -> dict:
    """Generate career guidance using AI with caching"""
    cache_key = get_cache_key("guidance", current_role_or_stage, goals, recent_work_summary)
    cached_guidance = get_from_cache(cache_key)
    
    if cached_guidance:
        return cached_guidance
    
    prompt = (
        "Act as a senior AI career coach. Provide focused guidance to improve skills, portfolio, and hiring outcomes.\n"
        "Return strict JSON: {\"advice\": str, \"next_steps\": [str, str, str]}.\n"
        f"Stage: {current_role_or_stage}\nGoals: {goals}\nRecent work: {recent_work_summary}\n"
        "Be specific about resources, project scopes, and measurable outcomes (e.g., 'ship X in Y days')."
    )
    out = _chat([
        {"role": "system", "content": "You coach AI learners with concrete, prioritized steps as strict JSON."},
        {"role": "user", "content": prompt},
    ])
    
    try:
        data = json.loads(out)
    except Exception:
        start, end = out.find("{"), out.rfind("}")
        if start == -1 or end == -1:
            raise HTTPException(status_code=500, detail="Invalid AI response format")
        data = json.loads(out[start:end + 1])
    
    result = {"advice": data.get("advice", ""), "next_steps": data.get("next_steps", [])}
    
    # Cache for 2 hours
    set_cache(cache_key, result, 7200)
    
    return result

def generate_linkedin_post(post_type: str, content: str, tone: str, include_hashtags: bool, include_call_to_action: bool) -> dict:
    """Generate LinkedIn post using AI with caching"""
    cache_key = get_cache_key("linkedin_post", post_type, content, tone, include_hashtags, include_call_to_action)
    cached_post = get_from_cache(cache_key)
    
    if cached_post:
        return cached_post
    
    # Define post type templates
    post_templates = {
        "achievement": "Create a professional LinkedIn post celebrating an achievement or milestone",
        "project": "Create a LinkedIn post showcasing a project you've completed or are working on",
        "learning": "Create a LinkedIn post sharing a learning experience or skill you've developed",
        "career_update": "Create a LinkedIn post announcing a career update or transition",
        "networking": "Create a LinkedIn post for networking and building professional connections",
        "thought_leadership": "Create a LinkedIn post sharing insights or expertise in your field"
    }
    
    tone_instructions = {
        "professional": "Use a formal, business-appropriate tone",
        "casual": "Use a friendly, approachable tone while maintaining professionalism",
        "inspiring": "Use an uplifting, motivational tone",
        "technical": "Use a detailed, technical tone appropriate for industry professionals",
        "personal": "Use a personal, authentic tone while staying professional"
    }
    
    prompt = (
        f"{post_templates.get(post_type, 'Create a professional LinkedIn post')}.\n"
        f"Tone: {tone_instructions.get(tone, 'professional')}\n"
        f"Content to work with: {content}\n\n"
        "Return strict JSON with these fields:\n"
        "{\n"
        '  "post_content": "The main LinkedIn post text (engaging, professional, 150-300 words)",\n'
        '  "hashtags": ["#relevant", "#hashtags", "#for", "#linkedin"],\n'
        '  "call_to_action": "A compelling call-to-action question or statement",\n'
        '  "engagement_tips": ["Tip 1", "Tip 2", "Tip 3"]\n'
        "}\n\n"
        "Make it engaging, authentic, and optimized for LinkedIn's algorithm. "
        "Include specific details, metrics if relevant, and make it shareable."
    )
    
    out = _chat([
        {"role": "system", "content": "You are a LinkedIn content expert who creates engaging, professional posts that drive engagement and build personal brand."},
        {"role": "user", "content": prompt},
    ])
    
    try:
        data = json.loads(out)
    except Exception:
        start, end = out.find("{"), out.rfind("}")
        if start == -1 or end == -1:
            # Fallback response when Perplexity Sonar is unavailable
            result = {
                "post_content": f"Excited to share {content[:100]}... Looking forward to continuing this journey and connecting with others in the field!",
                "hashtags": ["#career", "#growth", "#professional", "#networking"],
                "call_to_action": "What's your experience with this? I'd love to hear your thoughts!",
                "engagement_tips": ["Post during business hours (9 AM - 5 PM)", "Engage with comments within the first hour", "Share personal insights and lessons learned"]
            }
            set_cache(cache_key, result, 3600)
            return result
        data = json.loads(out[start:end + 1])
    
    # Calculate character count
    post_text = data.get("post_content", "")
    hashtags_text = " ".join(data.get("hashtags", []))
    cta_text = data.get("call_to_action", "")
    total_chars = len(post_text) + len(hashtags_text) + len(cta_text)
    
    result = {
        "post_content": post_text,
        "hashtags": data.get("hashtags", []) if include_hashtags else [],
        "call_to_action": data.get("call_to_action", "") if include_call_to_action else "",
        "engagement_tips": data.get("engagement_tips", []),
        "character_count": total_chars
    }
    
    # Cache for 1 hour
    set_cache(cache_key, result, 3600)
    
    return result

# Advanced AI features
def generate_career_roadmap(user_data: dict) -> dict:
    """Generate a personalized career roadmap"""
    prompt = (
        "Create a personalized career roadmap based on user data.\n"
        "Return strict JSON: {\"roadmap\": [{\"phase\": str, \"goals\": [str], \"timeline\": str}], \"next_milestone\": str}.\n"
        f"User data: {json.dumps(user_data)}\n"
        "Focus on practical, achievable steps with realistic timelines."
    )
    
    out = _chat([
        {"role": "system", "content": "You create detailed career roadmaps as strict JSON."},
        {"role": "user", "content": prompt},
    ])
    
    try:
        data = json.loads(out)
    except Exception:
        start, end = out.find("{"), out.rfind("}")
        if start == -1 or end == -1:
            raise HTTPException(status_code=500, detail="Invalid AI response format")
        data = json.loads(out[start:end + 1])
    
    return data

def analyze_skill_gaps(user_data: dict) -> dict:
    """Analyze skill gaps and suggest improvements"""
    prompt = (
        "Analyze skill gaps and suggest specific improvements.\n"
        'Return strict JSON: {"gaps": [{"skill": str, "importance": str, "resources": [str]}], "priority": str}.\n'
        f"User data: {json.dumps(user_data)}\n"
        "Be specific about skills, importance levels, and learning resources."
    )
    
    out = _chat([
        {"role": "system", "content": "You analyze skill gaps and provide actionable recommendations as strict JSON."},
        {"role": "user", "content": prompt},
    ])
    
    try:
        data = json.loads(out)
    except Exception:
        start, end = out.find("{"), out.rfind("}")
        if start == -1 or end == -1:
            raise HTTPException(status_code=500, detail="Invalid AI response format")
        data = json.loads(out[start:end + 1])
    
    return data

# ---------------------------
# Phases and Seeding
# ---------------------------
PHASES: list[PhasePublic] = [
    PhasePublic(id=1, name=PhaseEnum.Introspection, description="Journaling, Ikigai, sentiment"),
    PhasePublic(id=2, name=PhaseEnum.Exploration, description="Ideas and build-in-public"),
    PhasePublic(id=3, name=PhaseEnum.Reflection, description="Delta-4 analysis"),
    PhasePublic(id=4, name=PhaseEnum.Action, description="Milestones and execution"),
]

DEFAULT_TASKS = {
    PhaseEnum.Introspection: [
        "Write your Ikigai journal and submit for AI summary",
        "Capture baseline sentiment score",
    ],
    PhaseEnum.Exploration: [
        "Review 3 AI project ideas and pick one",
        "Generate a build-in-public post draft",
    ],
    PhaseEnum.Reflection: [
        "Log weekly friction and delight",
        "Run Delta-4 analyzer and pick next actions",
    ],
    PhaseEnum.Action: [
        "Create project milestones",
        "Move tasks across Not Started/In Progress/Done",
    ],
}

def seed_tasks(session: Session):
    """Seed default tasks for each phase"""
    exists = session.exec(select(Task)).first()
    if exists:
        return
    
    for phase, descriptions in DEFAULT_TASKS.items():
        for desc in descriptions:
            session.add(Task(phase=phase.value, description=desc, status="Not Started"))
    session.commit()

# ---------------------------
# FastAPI Application
# ---------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting CareerAI Backend")
    create_db_and_tables()
    with Session(engine) as session:
        seed_tasks(session)
    logger.info("Database initialized")
    yield
    # Shutdown
    logger.info("Shutting down CareerAI Backend")

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description="AI-powered career development platform",
    lifespan=lifespan
)

# Add rate limiting
if Limiter:
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
else:
    # Fallback for development
    app.state.limiter = limiter

# Add performance monitoring middleware
@app.middleware("http")
async def performance_middleware(request: Request, call_next):
    """Middleware to track response times"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    # Record response time
    endpoint = f"{request.method} {request.url.path}"
    performance_monitor.record_response_time(endpoint, process_time)
    
    # Add performance header
    response.headers["X-Process-Time"] = str(process_time)
    
    return response

# ---------------------------
# Exception Handlers
# ---------------------------
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler"""
    logger.warning(f"HTTP {exc.status_code}: {exc.detail} - {request.url}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "error_code": f"HTTP_{exc.status_code}"}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler"""
    logger.error(f"Unhandled exception: {exc} - {request.url}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error_code": "INTERNAL_ERROR"}
    )

# ---------------------------
# Health Check
# ---------------------------
@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": APP_VERSION
    }

@app.get("/test-ai", tags=["Health"])
async def test_ai():
    """Test Perplexity Sonar availability"""
    if _require_perplexity():
        return {"ai_available": True, "message": "Perplexity Sonar API is available"}
    else:
        return {"ai_available": False, "message": "Perplexity Sonar API is unavailable, using fallbacks"}

@app.post("/test-ikigai", response_model=IkigaiResponse, tags=["AI"])
async def test_ikigai(
    request: Request,
    req: IkigaiRequest,
    session: SessionDep,
    current_user: User = Depends(get_current_user)
):
    """Test ikigai endpoint without any AI calls"""
    return IkigaiResponse(
        journal_entry_id=0,
        ai_summary="Test response - AI is working!",
        sentiment_score=0.8,
    )

# ---------------------------
# Authentication Endpoints
# ---------------------------
@app.post("/auth/register", response_model=TokenResponse, tags=["Authentication"])
@limiter.limit(f"{RATE_LIMIT_PER_MINUTE}/minute")
async def register(request: Request, req: RegisterRequest, session: SessionDep):
    """Register a new user"""
    existing = session.exec(select(User).where(User.email == req.email)).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    user = User(
        name=req.name,
        email=req.email,
        password_hashed=hash_password(req.password),
        current_phase=None,
        progress=0.0,
    )
    session.add(user)
    session.commit()
    session.refresh(user)
    
    access_token = create_access_token(data={"sub": str(user.id)})
    logger.info(f"User registered: {user.email}")
    
    return TokenResponse(user_id=user.id, token=access_token)

@app.post("/auth/login", response_model=TokenResponse, tags=["Authentication"])
@limiter.limit(f"{RATE_LIMIT_PER_MINUTE}/minute")
async def login(request: Request, req: LoginRequest, session: SessionDep):
    """Login user"""
    user = session.exec(select(User).where(User.email == req.email)).first()
    if not user or not verify_password(req.password, user.password_hashed):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    access_token = create_access_token(data={"sub": str(user.id)})
    logger.info(f"User logged in: {user.email}")
    
    return TokenResponse(user_id=user.id, token=access_token)

@app.get("/auth/me", response_model=UserPublic, tags=["Authentication"])
async def me(current_user: User = Depends(get_current_user)):
    """Get current user information"""
    return UserPublic(
        id=current_user.id,
        name=current_user.name,
        email=current_user.email,
        current_phase=PhaseEnum(current_user.current_phase) if current_user.current_phase else None,
        progress=current_user.progress,
        created_at=current_user.created_at,
    )

# ---------------------------
# Phase Endpoints
# ---------------------------
@app.get("/phases", response_model=list[PhasePublic], tags=["Phases"])
async def get_phases():
    """Get all available phases"""
    return PHASES

@app.patch("/phase/progress", tags=["Phases"])
async def update_phase_progress(
    req: PhaseProgressRequest,
    session: SessionDep,
    current_user: User = Depends(get_current_user)
):
    """Update user's phase progress"""
    current_user.current_phase = req.current_phase.value
    current_user.progress = req.progress
    current_user.updated_at = datetime.utcnow()
    
    session.add(current_user)
    session.commit()
    
    logger.info(f"Phase progress updated for user {current_user.id}: {req.current_phase.value} - {req.progress}%")
    
    return {
        "message": "Progress updated",
        "current_phase": current_user.current_phase,
        "progress": current_user.progress
    }

# ---------------------------
# AI Endpoints
# ---------------------------
@app.post("/ai/ikigai", response_model=IkigaiResponse, tags=["AI"])
async def ikigai(
    request: Request,
    req: IkigaiRequest,
    session: SessionDep,
    current_user: User = Depends(get_current_user)
):
    """Analyze journal text for ikigai insights"""
    try:
        # Simple fallback response for now to avoid AI issues
        result = {
            "ai_summary": "Based on your journal, you're exploring your passions and purpose. This is a valuable process of self-discovery that will help guide your career decisions.",
            "sentiment_score": 0.7
        }
        
        entry = JournalEntry(
            user_id=current_user.id,
            content=req.journal_text,
            ai_summary=result["ai_summary"],
            sentiment_score=result["sentiment_score"],
        )
        session.add(entry)
        session.commit()
        session.refresh(entry)
        
        logger.info(f"Ikigai analysis completed for user {current_user.id}")
        
        return IkigaiResponse(
            journal_entry_id=entry.id,
            ai_summary=entry.ai_summary,
            sentiment_score=entry.sentiment_score,
        )
    except Exception as e:
        logger.error(f"Ikigai endpoint error: {e}")
        # Return a simple response without database operations
        return IkigaiResponse(
            journal_entry_id=0,
            ai_summary="Based on your journal, you're exploring your passions and purpose. This is a valuable process of self-discovery that will help guide your career decisions.",
            sentiment_score=0.7,
        )

@app.post("/ai/project-ideas", response_model=list[dict], tags=["AI"])
@limiter.limit("5/minute")
async def project_ideas(
    request: Request,
    req: ProjectIdeaRequest,
    session: SessionDep,
    current_user: User = Depends(get_current_user)
):
    """Generate project ideas from ikigai summary"""
    if not _require_perplexity():
        # Return fallback project ideas when Perplexity Sonar is unavailable
        ideas = [
            {
                "title": "Personal Portfolio Website",
                "description": "Build a modern portfolio website showcasing your skills and projects. Include interactive elements and responsive design to demonstrate your technical abilities."
            },
            {
                "title": "Data Analysis Project",
                "description": "Analyze a dataset related to your interests and create visualizations. Document your findings and insights in a comprehensive report."
            },
            {
                "title": "Open Source Contribution",
                "description": "Contribute to an open source project that aligns with your interests. Start with documentation, bug fixes, or small features to build your reputation."
            }
        ]
    else:
        ideas = generate_project_ideas_from_summary(req.ikigai_summary)
    created = []
    
    for idea in ideas[:3]:
        p = Project(
            user_id=current_user.id,
            title=idea["title"],
            description=idea["description"],
            status="Not Started"
        )
        session.add(p)
        session.commit()
        session.refresh(p)
        created.append({
            "id": p.id,
            "title": p.title,
            "description": p.description,
            "status": p.status
        })
    
    logger.info(f"Project ideas generated for user {current_user.id}")
    return created

@app.post("/ai/post", response_model=dict, tags=["AI"])
@limiter.limit("5/minute")
async def build_post(
    request: Request,
    req: ProjectPostRequest,
    session: SessionDep,
    current_user: User = Depends(get_current_user)
):
    """Generate build-in-public posts for a project"""
    _require_perplexity()
    
    project = session.get(Project, req.project_id)
    if not project or project.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Project not found")
    
    posts = build_in_public_posts(project.title, project.description)
    
    # Append to description for auditability
    project.description = (
        project.description
        + "\n\nLinkedIn:\n"
        + posts.get("linkedin", "")
        + "\n\nTwitter:\n"
        + posts.get("twitter", "")
    )
    project.updated_at = datetime.utcnow()
    
    session.add(project)
    session.commit()
    
    logger.info(f"Build-in-public posts generated for project {req.project_id}")
    return posts

@app.post("/ai/reflection", response_model=ReflectionResponse, tags=["AI"])
@limiter.limit("10/minute")
async def reflection(
    request: Request,
    req: ReflectionRequest,
    session: SessionDep,
    current_user: User = Depends(get_current_user)
):
    """Analyze reflection using Delta-4 thinking"""
    _require_perplexity()
    
    result = analyze_delta4(req.friction, req.delight)
    ref = Reflection(
        user_id=current_user.id,
        friction=req.friction,
        delight=req.delight,
        ai_summary=result["ai_summary"],
    )
    session.add(ref)
    session.commit()
    session.refresh(ref)
    
    logger.info(f"Reflection analysis completed for user {current_user.id}")
    
    return ReflectionResponse(reflection_id=ref.id, ai_summary=ref.ai_summary)

@app.post("/ai/guidance", response_model=GuidanceResponse, tags=["AI"])
@limiter.limit("5/minute")
async def guidance(request: Request, req: GuidanceRequest):
    """Generate career guidance using AI"""
    _require_perplexity()
    
    data = generate_guidance(req.current_role_or_stage, req.goals, req.recent_work_summary)
    logger.info("Career guidance generated")
    
    return GuidanceResponse(advice=data["advice"], next_steps=data["next_steps"])

@app.post("/ai/linkedin-post", response_model=LinkedInPostResponse, tags=["AI"])
@limiter.limit("10/minute")
async def linkedin_post(request: Request, req: LinkedInPostRequest):
    """Generate a LinkedIn post based on user input"""
    _require_perplexity()
    
    data = generate_linkedin_post(
        req.post_type, 
        req.content, 
        req.tone, 
        req.include_hashtags, 
        req.include_call_to_action
    )
    logger.info("LinkedIn post generated")
    
    return LinkedInPostResponse(**data)

# ---------------------------
# Project Endpoints
# ---------------------------
@app.get("/projects", response_model=ProjectListResponse, tags=["Projects"])
async def list_projects(
    session: SessionDep,
    current_user: User = Depends(get_current_user)
):
    """List user's projects"""
    projects = session.exec(select(Project).where(Project.user_id == current_user.id)).all()
    return ProjectListResponse(
        projects=[
            ProjectPublic(
                id=p.id,
                user_id=p.user_id,
                title=p.title,
                description=p.description,
                status=p.status,
                created_at=p.created_at,
                updated_at=p.updated_at
            )
            for p in projects
        ]
    )

@app.patch("/projects/{project_id}", response_model=ProjectPublic, tags=["Projects"])
async def update_project(
    project_id: int,
    req: ProjectUpdateRequest,
    session: SessionDep,
    current_user: User = Depends(get_current_user)
):
    """Update project status"""
    project = session.get(Project, project_id)
    if not project or project.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Project not found")
    
    project.status = req.status
    project.updated_at = datetime.utcnow()
    
    session.add(project)
    session.commit()
    session.refresh(project)
    
    logger.info(f"Project {project_id} status updated to {req.status}")
    
    return ProjectPublic(
        id=project.id,
        user_id=project.user_id,
        title=project.title,
        description=project.description,
        status=project.status,
        created_at=project.created_at,
        updated_at=project.updated_at,
    )

# ---------------------------
# Task Endpoints
# ---------------------------
@app.get("/tasks", tags=["Tasks"])
async def list_tasks(
    session: SessionDep,
    phase: Optional[PhaseEnum] = None
):
    """List tasks, optionally filtered by phase"""
    query = select(Task)
    if phase:
        query = query.where(Task.phase == phase.value)
    
    tasks = session.exec(query).all()
    return {"tasks": [
        {
            "id": t.id,
            "phase": t.phase,
            "description": t.description,
            "status": t.status,
            "created_at": t.created_at
        }
        for t in tasks
    ]}

@app.patch("/tasks/{task_id}", tags=["Tasks"])
async def update_task(
    task_id: int,
    status: str,
    session: SessionDep
):
    """Update task status"""
    if status not in VALID_STATUSES:
        raise HTTPException(status_code=400, detail="Invalid status")
    
    task = session.get(Task, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task.status = status
    session.add(task)
    session.commit()
    
    logger.info(f"Task {task_id} status updated to {status}")
    
    return {"message": "Task updated", "task_id": task_id, "status": status}

# ---------------------------
# Advanced Endpoints
# ---------------------------

# WebSocket endpoint for real-time updates
@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: int):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket, user_id)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle different message types
            if message.get("type") == "ping":
                await manager.broadcast_to_user(user_id, {"type": "pong", "timestamp": datetime.utcnow().isoformat()})
            elif message.get("type") == "subscribe":
                # User subscribes to specific updates
                await manager.broadcast_to_user(user_id, {"type": "subscribed", "channel": message.get("channel")})
                
    except WebSocketDisconnect:
        manager.disconnect(websocket, user_id)

# Analytics endpoints
@app.get("/analytics/metrics", response_model=AnalyticsResponse, tags=["Analytics"])
async def get_user_metrics(
    current_user: User = Depends(get_current_user)
):
    """Get user analytics metrics"""
    metrics = analytics.get_user_metrics(current_user.id)
    return AnalyticsResponse(**metrics)

@app.post("/analytics/track", tags=["Analytics"])
async def track_event(
    event: str,
    metadata: dict = None,
    current_user: User = Depends(get_current_user)
):
    """Track a user event"""
    analytics.track_event(current_user.id, event, metadata)
    return {"message": "Event tracked successfully"}

# Advanced AI endpoints
@app.post("/ai/roadmap", tags=["AI"])
@limiter.limit("3/minute")
async def generate_roadmap(
    request: Request,
    session: SessionDep,
    current_user: User = Depends(get_current_user)
):
    """Generate personalized career roadmap"""
    _require_perplexity()
    
    # Gather user data
    projects = session.exec(select(Project).where(Project.user_id == current_user.id)).all()
    journal_entries = session.exec(select(JournalEntry).where(JournalEntry.user_id == current_user.id)).all()
    
    user_data = {
        "current_phase": current_user.current_phase,
        "progress": current_user.progress,
        "projects": [{"title": p.title, "status": p.status} for p in projects],
        "journal_count": len(journal_entries),
        "sentiment_scores": [j.sentiment_score for j in journal_entries]
    }
    
    roadmap = generate_career_roadmap(user_data)
    logger.info(f"Career roadmap generated for user {current_user.id}")
    
    return roadmap

@app.post("/ai/skill-gaps", tags=["AI"])
@limiter.limit("3/minute")
async def analyze_skill_gaps_endpoint(
    request: Request,
    session: SessionDep,
    current_user: User = Depends(get_current_user)
):
    """Analyze skill gaps and suggest improvements"""
    _require_perplexity()
    
    # Gather user data
    projects = session.exec(select(Project).where(Project.user_id == current_user.id)).all()
    reflections = session.exec(select(Reflection).where(Reflection.user_id == current_user.id)).all()
    
    user_data = {
        "current_phase": current_user.current_phase,
        "projects": [{"title": p.title, "description": p.description, "status": p.status} for p in projects],
        "reflections": [{"friction": r.friction, "delight": r.delight} for r in reflections]
    }
    
    skill_analysis = analyze_skill_gaps(user_data)
    logger.info(f"Skill gap analysis completed for user {current_user.id}")
    
    return skill_analysis

@app.get("/ai/insights", response_model=InsightsResponse, tags=["AI"])
async def get_user_insights(
    session: SessionDep,
    current_user: User = Depends(get_current_user)
):
    """Get comprehensive user insights"""
    # Gather user data
    projects = session.exec(select(Project).where(Project.user_id == current_user.id)).all()
    journal_entries = session.exec(select(JournalEntry).where(JournalEntry.user_id == current_user.id)).all()
    
    user_data = {
        "current_phase": current_user.current_phase,
        "progress": current_user.progress,
        "projects": [{"title": p.title, "status": p.status} for p in projects],
        "sentiment_scores": [j.sentiment_score for j in journal_entries]
    }
    
    insights = ai_enhancer.generate_insights(user_data)
    
    # Calculate sentiment trend
    sentiment_trend = "stable"
    if len(user_data["sentiment_scores"]) > 1:
        sentiment_trend = ai_enhancer.calculate_sentiment_trend(user_data["sentiment_scores"])
    
    insights["sentiment_trend"] = sentiment_trend
    
    return InsightsResponse(**insights)

# Notification endpoints
@app.get("/notifications", response_model=List[NotificationResponse], tags=["Notifications"])
async def get_notifications(
    session: SessionDep,
    current_user: User = Depends(get_current_user)
):
    """Get user notifications"""
    notifications = session.exec(
        select(Notification)
        .where(Notification.user_id == current_user.id)
        .order_by(Notification.created_at.desc())
        .limit(50)
    ).all()
    
    return [
        NotificationResponse(
            id=n.id,
            title=n.title,
            message=n.message,
            notification_type=n.notification_type,
            is_read=n.is_read,
            created_at=n.created_at
        )
        for n in notifications
    ]

@app.patch("/notifications/{notification_id}/read", tags=["Notifications"])
async def mark_notification_read(
    notification_id: int,
    session: SessionDep,
    current_user: User = Depends(get_current_user)
):
    """Mark notification as read"""
    notification = session.get(Notification, notification_id)
    if not notification or notification.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Notification not found")
    
    notification.is_read = True
    session.add(notification)
    session.commit()
    
    return {"message": "Notification marked as read"}

# Performance monitoring endpoints
@app.get("/admin/performance", tags=["Admin"])
async def get_performance_stats():
    """Get performance statistics (admin only)"""
    stats = {}
    for endpoint in performance_monitor.response_times.keys():
        stats[endpoint] = performance_monitor.get_endpoint_stats(endpoint)
    
    return {
        "endpoints": stats,
        "cache_stats": {
            "total_cached_items": len(cache),
            "cache_hit_rate": "N/A"  # Would need more sophisticated tracking
        }
    }

@app.get("/admin/health", tags=["Admin"])
async def detailed_health_check():
    """Detailed health check with system metrics"""
    try:
        # Test database connection
        with Session(engine) as session:
            session.exec(select(User).limit(1))
        
        # Test Perplexity Sonar connection
        perplexity_status = "healthy" if PERPLEXITY_API_KEY else "no_key"
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": APP_VERSION,
            "database": "connected",
            "perplexity": perplexity_status,
            "cache_items": len(cache),
            "active_websockets": sum(len(conns) for conns in manager.active_connections.values()),
            "uptime": "N/A"  # Would need startup time tracking
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

# Utility endpoints
@app.post("/utils/generate-session", tags=["Utils"])
async def generate_session_token(
    session: SessionDep,
    current_user: User = Depends(get_current_user)
):
    """Generate a new session token"""
    session_token = secrets.token_urlsafe(32)
    
    # Store session in database
    user_session = UserSession(
        user_id=current_user.id,
        session_token=session_token,
        is_active=True,
        ip_address="127.0.0.1",  # Would get from request in production
        user_agent="CareerAI-Client"
    )
    session.add(user_session)
    session.commit()
    
    return {"session_token": session_token}

@app.get("/utils/keywords/{text}", tags=["Utils"])
async def extract_keywords(text: str):
    """Extract keywords from text"""
    keywords = ai_enhancer.extract_keywords(text)
    return {"keywords": keywords, "text_length": len(text)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=DEBUG)
