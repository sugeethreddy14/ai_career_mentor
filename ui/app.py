import os
import streamlit as st
import requests
import fitz  # PyMuPDF for PDF text extraction (frontend helper)
import json
import io
import mimetypes

# Configuration
# Use BACKEND_URL env var (Render will set this in the Environment settings).
# Default falls back to your local backend for local development (preserves your
# existing local ports). When deployed on Render, set BACKEND_URL in the
# frontend service's environment to the backend's Render URL.
# Locally this keeps the backend at http://localhost:7311 (as in run.txt).
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:7311")

st.set_page_config(layout="wide", page_title="AI Career Mentor")
st.title("AI Career Mentor")

# --- Input Area ---
col1, col2 = st.columns([1, 1])

with col1:
    resume_input = st.text_area("Paste your resume or LinkedIn summary here:", height=300, key="text_input")

with col2:
    uploaded_file = st.file_uploader("Or upload your resume PDF/DOCX", type=["pdf", "docx"], key="file_uploader")


# NOTE: PDF/DOCX extraction is moved to the backend (main.py) for stability, 
# but this function is kept for local testing if needed.
def extract_text_from_pdf(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def create_multipart_payload(text_input, uploaded_file):
    """
    Creates the payload dictionary for a multipart/form-data request.
    This is necessary because the backend now expects form fields and files.
    """
    data = {}
    files = {}

    # 1. Handle Text Input (Form field)
    if text_input and text_input.strip():
        data['resume_text'] = (None, text_input) # (filename, content, media_type) format for form fields

    # 2. Handle File Upload (File field)
    if uploaded_file is not None:
        file_mime = mimetypes.guess_type(uploaded_file.name)[0] or 'application/octet-stream'
        
        # The 'file' key must match the 'file' parameter name in FastAPI's @app.post
        files['file'] = (uploaded_file.name, uploaded_file.read(), file_mime)
        
    return data, files


if st.button("Analyze Profile", use_container_width=True, type="primary"):
    
    data_payload, files_payload = create_multipart_payload(resume_input, uploaded_file)
    
    if not data_payload and not files_payload:
        st.error("Please enter text or upload a file to analyze.")
    else:
        with st.spinner("Analyzing profile and generating detailed roadmap..."):
            try:
                # --- SEND MULTIPART REQUEST ---
                # requests.post automatically uses multipart/form-data when 'files' is provided.
                # If only 'data' is provided, it uses standard form encoding.
                response = requests.post(
                    f"{BACKEND_URL}/analyze-profile",
                    data=data_payload,
                    files=files_payload,
                    timeout=120, 
                )
                
                st.markdown("---")
                
                if response.status_code == 200:
                    data = response.json()

                    # =======================================================
                    # DIAGNOSTICS
                    # =======================================================
                    with st.expander("🔍 Raw Backend Response (Diagnostics)"):
                        st.json(data)
                        if "suggested_roles" not in data or not data["suggested_roles"]:
                            st.error("🚨 Backend returned an empty or missing 'suggested_roles' key.")
                    # =======================================================

                    # --- Main Display Columns ---
                    result_col1, result_col2 = st.columns(2)
                    
                    # Extracted Skills
                    with result_col1:
                        st.subheader("🎯 Extracted Skills")
                        skills = data.get("skills", [])
                        st.success(", ".join(skills) if skills else "No skills found.")
                        
                        st.subheader("👤 Career Level")
                        level = data.get("level", "N/A")
                        st.markdown(f"**{level}**")

                        # New: Skill Proficiency Chart Data
                        skill_scores = data.get("skill_scores", {})
                        if skill_scores:
                            import pandas as pd
                            st.subheader("📊 Skill Proficiency Analysis")
                            df = pd.DataFrame(list(skill_scores.items()), columns=['Skill', 'Score'])
                            # Convert score from 1-10 to 0-100 for better chart display
                            df['Score'] = df['Score'].apply(lambda x: min(100, max(0, x * 10)))
                            st.bar_chart(df.set_index("Skill"), use_container_width=True)


                    # Suggested Roles and Alignment
                    with result_col2:
                        st.subheader("💡 Suggested Roles")
                        suggested_roles = data.get("suggested_roles", [])
                        if suggested_roles:
                            for role in suggested_roles:
                                st.write(f"**• {role}**")
                        else:
                            st.warning("No suggested roles found.")

                        # New: Role Alignment Gaps (Actionable items)
                        gaps = data.get("critical_knowledge_gaps", [])
                        if gaps:
                            st.subheader("🚧 Critical Knowledge Gaps")
                            for gap in gaps:
                                st.write(f"- ⚠️ {gap}")


                    st.markdown("---")

                    # --- Roadmap & Projects (High Quality LLM Output) ---
                    roadmap_col, projects_col = st.columns(2)
                    
                    with roadmap_col:
                        st.subheader("🗺️ Detailed Learning Roadmap")
                        roadmap = data.get("learning_roadmap", ["No roadmap generated."])
                        for step in roadmap:
                            st.markdown(f"**•** {step}")
                            
                    with projects_col:
                        st.subheader("🚀 Advanced Mini-Projects")
                        projects = data.get("projects", ["No projects suggested."])
                        for proj in projects:
                            st.markdown(f"**•** {proj}")

                else:
                    error_detail = response.json().get('detail', 'Unknown error.')
                    st.error(f"Analysis failed with status code {response.status_code}. Detail: {error_detail}")
            
            except requests.exceptions.RequestException as e:
                st.error(f"Could not connect to backend server at {BACKEND_URL}. Ensure your backend service is running on the correct port. Error: {e}")
