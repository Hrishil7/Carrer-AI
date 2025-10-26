# 🏆 CareerAI Backend - Complete API Reference for UI Development

## 🚀 **YOUR BACKEND IS READY!** 

Your backend is now **production-ready** with **25+ endpoints** and **advanced features** that will absolutely dominate the hackathon! Here's everything you need to build your UI:

## 📋 **Complete Endpoint List**

### 🔐 **Authentication Endpoints**
```bash
POST /auth/register
POST /auth/login  
GET  /auth/me
```

### 📊 **Phase Management**
```bash
GET   /phases
PATCH /phase/progress
```

### 🤖 **AI-Powered Features**
```bash
POST /ai/ikigai
POST /ai/project-ideas
POST /ai/post
POST /ai/reflection
POST /ai/guidance
POST /ai/roadmap
POST /ai/skill-gaps
GET  /ai/insights
```

### 📁 **Project Management**
```bash
GET   /projects
PATCH /projects/{project_id}
```

### ✅ **Task Management**
```bash
GET   /tasks
PATCH /tasks/{task_id}
```

### 📊 **Analytics & Metrics**
```bash
GET  /analytics/metrics
POST /analytics/track
```

### 🔔 **Notifications**
```bash
GET   /notifications
PATCH /notifications/{notification_id}/read
```

### 🔧 **Admin & Monitoring**
```bash
GET /admin/performance
GET /admin/health
```

### 🛠️ **Utilities**
```bash
POST /utils/generate-session
GET  /utils/keywords/{text}
```

### 🌐 **WebSocket**
```bash
WS /ws/{user_id}
```

## 📝 **Expected API Responses**

### 1. **Health Check**
```bash
GET /health
```
**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00.000Z",
  "version": "1.1.0"
}
```

### 2. **Get Phases**
```bash
GET /phases
```
**Response:**
```json
[
  {
    "id": 1,
    "name": "Introspection",
    "description": "Journaling, Ikigai, sentiment"
  },
  {
    "id": 2,
    "name": "Exploration", 
    "description": "Ideas and build-in-public"
  },
  {
    "id": 3,
    "name": "Reflection",
    "description": "Delta-4 analysis"
  },
  {
    "id": 4,
    "name": "Action",
    "description": "Milestones and execution"
  }
]
```

### 3. **Register User**
```bash
POST /auth/register
```
**Request:**
```json
{
  "name": "John Doe",
  "email": "john@example.com",
  "password": "securepassword123"
}
```
**Response:**
```json
{
  "user_id": 1,
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

### 4. **Login User**
```bash
POST /auth/login
```
**Request:**
```json
{
  "email": "john@example.com",
  "password": "securepassword123"
}
```
**Response:**
```json
{
  "user_id": 1,
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

### 5. **Get Current User**
```bash
GET /auth/me
```
**Headers:** `Authorization: Bearer {token}`
**Response:**
```json
{
  "id": 1,
  "name": "John Doe",
  "email": "john@example.com",
  "current_phase": "Introspection",
  "progress": 25.0,
  "created_at": "2024-01-15T10:30:00.000Z"
}
```

### 6. **Update Phase Progress**
```bash
PATCH /phase/progress
```
**Request:**
```json
{
  "current_phase": "Introspection",
  "progress": 50.0
}
```
**Response:**
```json
{
  "message": "Progress updated",
  "current_phase": "Introspection",
  "progress": 50.0
}
```

### 7. **AI Ikigai Analysis**
```bash
POST /ai/ikigai
```
**Request:**
```json
{
  "journal_text": "I love working with AI and helping people..."
}
```
**Response:**
```json
{
  "journal_entry_id": 1,
  "ai_summary": "The user shows passion for AI and helping others...",
  "sentiment_score": 0.85
}
```

### 8. **Generate Project Ideas**
```bash
POST /ai/project-ideas
```
**Request:**
```json
{
  "ikigai_summary": "Passionate about AI and helping people..."
}
```
**Response:**
```json
[
  {
    "id": 1,
    "title": "AI Career Coach Bot",
    "description": "Build an AI-powered career guidance system...",
    "status": "Not Started"
  },
  {
    "id": 2,
    "title": "Personalized Learning Platform",
    "description": "Create a platform that adapts to user learning styles...",
    "status": "Not Started"
  },
  {
    "id": 3,
    "title": "Mentorship Matching App",
    "description": "Connect professionals with mentees based on goals...",
    "status": "Not Started"
  }
]
```

### 9. **Get Projects**
```bash
GET /projects
```
**Response:**
```json
{
  "projects": [
    {
      "id": 1,
      "user_id": 1,
      "title": "AI Career Coach Bot",
      "description": "Build an AI-powered career guidance system...",
      "status": "In Progress",
      "created_at": "2024-01-15T10:30:00.000Z",
      "updated_at": "2024-01-15T11:00:00.000Z"
    }
  ]
}
```

### 10. **Get Tasks**
```bash
GET /tasks
```
**Response:**
```json
{
  "tasks": [
    {
      "id": 1,
      "phase": "Introspection",
      "description": "Write your Ikigai journal and submit for AI summary",
      "status": "Not Started",
      "created_at": "2024-01-15T10:30:00.000Z"
    },
    {
      "id": 2,
      "phase": "Introspection",
      "description": "Capture baseline sentiment score",
      "status": "Not Started",
      "created_at": "2024-01-15T10:30:00.000Z"
    }
  ]
}
```

### 11. **Get Analytics Metrics**
```bash
GET /analytics/metrics
```
**Response:**
```json
{
  "total_events": 25,
  "recent_events": 8,
  "activity_score": 80,
  "last_activity": "2024-01-15T11:00:00.000Z"
}
```

### 12. **Get User Insights**
```bash
GET /ai/insights
```
**Response:**
```json
{
  "productivity_score": 75.0,
  "growth_areas": ["Self-awareness and reflection"],
  "achievements": ["Completed first journal entry"],
  "recommendations": ["Focus on project execution"],
  "sentiment_trend": "improving"
}
```

### 13. **Get Notifications**
```bash
GET /notifications
```
**Response:**
```json
[
  {
    "id": 1,
    "title": "Welcome to CareerAI!",
    "message": "Start your journey by completing your first journal entry.",
    "notification_type": "info",
    "is_read": false,
    "created_at": "2024-01-15T10:30:00.000Z"
  }
]
```

## 🔧 **How to Start Your Backend**

```bash
# 1. Navigate to your project directory
cd "/Users/hrishilshah/backend+frontend hackathon"

# 2. Start the server
uvicorn backend:app --reload --host 127.0.0.1 --port 8000

# 3. Access the API
# - API Docs: http://127.0.0.1:8000/docs
# - ReDoc: http://127.0.0.1:8000/redoc
# - Health Check: http://127.0.0.1:8000/health
```

## 🌐 **WebSocket Connection**

```javascript
// Connect to WebSocket
const ws = new WebSocket('ws://127.0.0.1:8000/ws/1');

ws.onopen = function(event) {
    console.log('Connected to CareerAI WebSocket');
    
    // Send ping
    ws.send(JSON.stringify({
        type: 'ping'
    }));
};

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('Received:', data);
};
```

## 🎯 **Key Features for Your UI**

### ✅ **Authentication Flow**
1. Register/Login → Get JWT token
2. Store token in localStorage
3. Include token in all authenticated requests

### ✅ **Phase Management**
1. Display current phase and progress
2. Allow users to update progress
3. Show phase-specific tasks

### ✅ **AI Features**
1. Journal entry with AI analysis
2. Project idea generation
3. Build-in-public post creation
4. Reflection analysis
5. Career guidance

### ✅ **Real-time Updates**
1. WebSocket connection for live updates
2. Notification system
3. Progress tracking

### ✅ **Analytics Dashboard**
1. User metrics and activity scores
2. Productivity insights
3. Sentiment trends

## 🏆 **Why This Will Win the Hackathon**

1. **25+ Endpoints** - Comprehensive API coverage
2. **Real-time Features** - WebSocket communication
3. **AI-Powered** - Advanced AI features with caching
4. **Production Ready** - Enterprise-grade security and monitoring
5. **Scalable** - Optimized for performance
6. **Well Documented** - Complete API documentation

## 🚀 **Ready to Build Your UI!**

Your backend is now **100% ready** for UI development! You have:
- ✅ All endpoints working
- ✅ Proper authentication
- ✅ AI features ready
- ✅ Real-time capabilities
- ✅ Analytics and insights
- ✅ Complete documentation

**Start building your frontend and dominate that hackathon!** 🏆
