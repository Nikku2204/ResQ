# resQ AI

![ResQ AI Logo](icon.png)

## Empowering Personal Safety Through Artificial Intelligence

ResQ AI is a comprehensive personal safety application designed to empower users with proactive safety tools while navigating both physical environments and digital interactions. Unlike traditional safety apps that only react after incidents occur, ResQ focuses on preventative measures, helping users identify potential threats before they become dangers.

## Core Features

### 🔍 Advanced Profile Verification
Verify the identity of individuals met online with our multi-layered profile verification system:
- Face detection and matching against fugitive databases (FBI & Interpol)
- Reverse image search for comprehensive web presence analysis
- AI-powered risk assessment and recommendations using GPT-4 Turbo
- Clear, actionable safety guidance based on verification results

### 🗺️ Location Safety Intelligence
Make informed decisions about where you go with real-time safety insights:
- Interactive safety heat maps showing risk levels across geographic areas
- AI-generated safety scores (0-100) with corresponding risk classifications
- Time-of-day adjusted safety assessments (Morning, Afternoon, Evening, Night)
- Incident visualization with detailed information about nearby safety concerns
- Custom safety routes to navigate through safer areas

### 💧 Hydrate: Smart Emergency Response
Our innovative emergency response system disguised as a hydration reminder app:
- Discreet safety check-ins that appear as regular hydration reminders
- Customizable check-in schedules (30min, 1hr, 2hr intervals)
- One-tap responses to confirm safety or trigger emergency protocols
- Automatic notifications to emergency contacts with your location when needed
- Silent audio recording for evidence collection in emergency situations

## Technology Stack

### AI & Machine Learning
- OpenAI GPT-4 Turbo for intelligent safety analysis and recommendations
- DeepFace (VGG-Face Model) for facial recognition and matching
- Google Cloud Vision API for image analysis and face detection
- Custom geospatial analysis algorithms for safety scoring

### Frontend
- HTML5, CSS3, JavaScript for responsive user interfaces
- Leaflet.js for interactive mapping and heat map visualization
- Dynamic data visualization for safety metrics presentation

### Backend
- Flask (Python) for API endpoints and server-side processing
- RESTful architecture for service integration
- Concurrent processing for image comparison and verification

### Data Sources
- Public safety data from municipal police departments
- Geographic information and mapping services
- Profile authentication through multiple verification methods

## Getting Started

### Prerequisites
- Python 3.8+
- Flask and associated dependencies
- API keys for: Google Cloud Vision, OpenAI, Scrapingdog, Imgur
- Access to safety data sources (e.g., Seattle Crime Data API)

### Installation

1. Clone the repository
```bash
git clone https://github.com/Nikku2204/ResQ.git
cd ResQ
