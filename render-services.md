Render service configuration (copy/paste)

Backend: ai-career-mentor-backend
- Type: Web Service
- Environment: Python 3.x
- Build Command: pip install -r requirements.txt
- Start Command: uvicorn app.main:app --host 0.0.0.0 --port 10000
- Health check: GET / (optional)

Frontend: ai-career-mentor-ui
- Type: Web Service
- Environment: Python 3.x
- Build Command: pip install -r requirements.txt
- Start Command: streamlit run ui/app.py --server.port=10000 --server.address=0.0.0.0

Environment variables to set (per-service)
- Backend (ai-career-mentor-backend):
  - LLM_API_KEY  # required for LLM calls
  - JOB_API_KEY  # optional external job API
  - MODEL_NAME (optional)
  - VECTOR_DB_PATH (optional)

- Frontend (ai-career-mentor-ui):
  - BACKEND_URL  # set this to the backend's public URL (e.g. https://ai-career-mentor-backend.onrender.com)

Notes:
- After deploying the backend, copy its public URL and add it as `BACKEND_URL` in the frontend service's Environment tab.
- If builds fail due to large ML dependencies (torch/transformers/faiss), consider using a lighter `requirements.txt` for Render that omits heavy ML libraries and only includes runtime packages (fastapi, uvicorn, streamlit, requests, python-dotenv, python-multipart)."}