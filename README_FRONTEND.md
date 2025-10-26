# 🧭 CareerAI Frontend

A beautiful, modern Streamlit frontend for the CareerAI application.

## ✨ Features

### 🎨 **Modern UI Design**
- **Glassmorphism Design**: Beautiful glass-like cards with blur effects
- **Gradient Backgrounds**: Stunning gradient backgrounds and animations
- **Responsive Layout**: Works perfectly on all screen sizes
- **Interactive Elements**: Hover effects, animations, and smooth transitions

### 📊 **Dashboard**
- **Real-time Metrics**: Activity score, total events, completed projects
- **Progress Tracking**: Visual progress bars and status indicators
- **Recent Activity**: Latest insights and user activity
- **User Profile**: Complete user information and progress

### 🔍 **Introspection Tab**
- **Ikigai Journal**: Discover your purpose through guided reflection
- **AI Analysis**: Get AI-powered insights about your journal entries
- **Sentiment Analysis**: Visual sentiment scoring
- **Keyword Extraction**: Identify key themes in your writing

### 💡 **Exploration Tab**
- **AI Project Ideas**: Generate personalized project suggestions
- **Resume-Ready Projects**: Scoped, actionable project ideas
- **Status Tracking**: Track project progress and completion

### 🤔 **Reflection Tab**
- **Delta-4 Framework**: Structured reflection on friction and delight
- **AI Insights**: Get actionable advice from your reflections
- **Action Items**: Clear next steps based on your analysis

### 📈 **Action Tab**
- **Phase Management**: Track your career development phases
- **Progress Updates**: Update your current phase and progress
- **Visual Progress**: Beautiful progress bars and status indicators

### 🚀 **Projects Tab**
- **Project Management**: View and manage all your projects
- **Status Updates**: Update project status (Not Started, In Progress, Done)
- **Project Details**: View project descriptions and current status

### 🎯 **Coach Tab**
- **Personalized Guidance**: Get AI-powered career advice
- **Goal Setting**: Define and track your career goals
- **Next Steps**: Clear action items for your career journey
- **Resource Recommendations**: Get suggested resources and tools

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Backend running on http://127.0.0.1:8000

### Installation

1. **Clone and navigate to the project:**
   ```bash
   cd "backend+frontend hackathon"
   ```

2. **Run the frontend:**
   ```bash
   ./run_frontend.sh
   ```

3. **Or run manually:**
   ```bash
   # Create virtual environment
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Start the frontend
   streamlit run app.py
   ```

4. **Open your browser:**
   - Frontend: http://localhost:8501
   - Backend API: http://127.0.0.1:8000

## 🎨 Design Features

### **Color Scheme**
- **Primary**: #7C5CFC (Purple)
- **Accent**: #22D3EE (Cyan)
- **Success**: #10B981 (Green)
- **Warning**: #F59E0B (Amber)
- **Error**: #EF4444 (Red)

### **UI Components**
- **Glass Cards**: Blurred background with transparency
- **Status Badges**: Color-coded status indicators
- **Progress Bars**: Animated progress visualization
- **Metric Cards**: Beautiful data visualization
- **Interactive Buttons**: Hover effects and animations

### **Animations**
- **Fade In**: Smooth page load animations
- **Hover Effects**: Interactive element animations
- **Loading Spinners**: Beautiful loading indicators
- **Progress Animations**: Smooth progress bar transitions

## 🔧 Configuration

### Environment Variables
Create a `.env` file in the project root:

```env
BACKEND_URL=http://127.0.0.1:8000
```

### Customization
- **Colors**: Modify the color variables in the CSS section
- **Layout**: Adjust the column layouts and spacing
- **Components**: Customize the glass effects and animations

## 📱 Responsive Design

The frontend is fully responsive and works on:
- **Desktop**: Full feature set with optimal layout
- **Tablet**: Adapted layout with touch-friendly controls
- **Mobile**: Mobile-optimized interface

## 🔗 Backend Integration

The frontend connects to the backend via HTTP API calls:
- **Authentication**: User registration and login
- **Data Fetching**: Real-time data from the backend
- **AI Features**: All AI-powered features
- **Analytics**: User metrics and insights

## 🎯 User Experience

### **Authentication Flow**
1. **Registration**: Create new account with validation
2. **Login**: Secure authentication with error handling
3. **Profile**: View user information and progress
4. **Logout**: Clean session management

### **Error Handling**
- **Connection Errors**: Graceful handling of backend connectivity
- **Validation Errors**: Clear error messages for user input
- **API Errors**: User-friendly error messages
- **Loading States**: Beautiful loading indicators

### **Success Feedback**
- **Toast Notifications**: Success messages for actions
- **Visual Feedback**: Status updates and progress indicators
- **Real-time Updates**: Live data refresh

## 🚀 Performance

- **Fast Loading**: Optimized CSS and JavaScript
- **Efficient API Calls**: Minimal and targeted requests
- **Caching**: Smart data caching for better performance
- **Responsive**: Smooth interactions on all devices

## 🎨 Customization

### **Themes**
The frontend uses a modern dark theme with:
- **Dark Background**: Easy on the eyes
- **High Contrast**: Excellent readability
- **Accent Colors**: Vibrant highlights
- **Glass Effects**: Modern aesthetic

### **Components**
All components are customizable:
- **Cards**: Adjust blur, transparency, and colors
- **Buttons**: Modify gradients and hover effects
- **Status Badges**: Change colors and styling
- **Progress Bars**: Customize appearance and animation

## 🔧 Development

### **File Structure**
```
app.py                 # Main Streamlit application
requirements.txt       # Python dependencies
run_frontend.sh       # Startup script
README_FRONTEND.md    # This documentation
```

### **Key Functions**
- `api()`: HTTP client for backend communication
- `toast()`: Success/error notifications
- `status_badge()`: Status indicator components
- `need_login()`: Authentication requirement message
- `success_message()`: Success feedback
- `error_message()`: Error feedback
- `info_message()`: Information messages

## 🎯 Next Steps

1. **Start the backend**: `./start_server.sh`
2. **Start the frontend**: `./run_frontend.sh`
3. **Open browser**: http://localhost:8501
4. **Register/Login**: Create your account
5. **Explore features**: Try all the tabs and features

## 🏆 Hackathon Ready

This frontend is designed to win hackathons with:
- **Modern Design**: Eye-catching and professional
- **Full Functionality**: All backend features integrated
- **Great UX**: Smooth and intuitive user experience
- **Responsive**: Works on all devices
- **Fast**: Optimized for performance

Your CareerAI application is now ready to impress judges and users! 🚀
