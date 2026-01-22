# 🧠 AI Career Mentor

### **AI-Driven Resume Analysis & Career Guidance Platform**

*A production-ready, full-stack multi-agent LLM system developed by  
**Sugeeth Reddy***

---

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-red)
![Gemini](https://img.shields.io/badge/Gemini-LLM-yellow)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Database-purple)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue)
![Render](https://img.shields.io/badge/Render-Deployed-success)
![License](https://img.shields.io/badge/License-MIT-blue)

![Issues](https://img.shields.io/github/issues/Tapasvi5fires/AI-CAREER-MENTOR)
![Forks](https://img.shields.io/github/forks/Tapasvi5fires/AI-CAREER-MENTOR)
![Stars](https://img.shields.io/github/stars/Tapasvi5fires/AI-CAREER-MENTOR)
![Last Commit](https://img.shields.io/github/last-commit/Tapasvi5fires/AI-CAREER-MENTOR)

---

## 🔗 Live Deployments

- **Frontend Application:** https://ai-career-mentor-frontend-2u2q.onrender.com  
- **Backend API:** https://ai-career-mentor-backend-2u2q.onrender.com  

---

## ✨ Core Capabilities

### 🔍 Resume Intelligence Extraction (GeminiAgent)

Extracts technical skills, soft skills, experience summaries, education,
certifications, readiness level, proficiency scores, and knowledge gaps,
producing a structured JSON profile.

---

### 🧠 Advanced Content Enrichment (DataEnrichmentAgent)

Generates personalized learning roadmaps, skill improvement plans,
career recommendations, and advanced project ideas using deep reasoning.

---

### 📊 Visualization Metrics

Provides radar charts, gap analysis, domain fit indicators, and strength matrices.

---

### 🔁 Reliability & Stability

Includes automatic retries, exponential backoff, and graceful handling
of timeouts, rate limits, and service errors.

---

## 🏗️ Architecture

User → Streamlit UI → FastAPI Backend → LLM Agents → Unified JSON → Visualizations

---

## 📂 Project Structure

```
AI-CAREER-MENTOR/
├── app/
│   ├── main.py
│   ├── agents/
│   │   ├── gemini_agent.py
│   │   ├── enrichment_agent.py
│   │   ├── mentor_agent.py
│   └── services/
│       ├── rag_service.py
├── ui/
│   └── app.py
├── start_backend.sh
├── start_frontend.sh
├── docker-compose.yml
├── requirements.txt
```

---

## ▶️ Running Locally

**Backend**
```
uvicorn app.main:app --reload --port 7311
```

**Frontend**
```
streamlit run ui/app.py --server.port=7312
```

---

## ☁️ Deployment (Render)

Environment variables:
```
LLM_API_KEY
BACKEND_URL
VECTOR_DB_PATH
JOB_API_KEY
```

---

## 🔥 Why This Project Stands Out

- Multi-agent LLM architecture  
- Full-stack AI engineering  
- Resume parsing pipeline  
- RAG with FAISS  
- Dockerized & cloud deployed  
- Production-grade error handling  

---

## ✨ Credits

Built with passion by **Sugeeth Reddy** ❤️
