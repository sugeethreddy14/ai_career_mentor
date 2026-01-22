import os
import json
import io
from typing import Optional, Annotated

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import asyncio
import logging

# File handling libraries
from docx import Document # python-docx
import fitz # PyMuPDF

# Import Agent dependencies
from app.agents.llm_profile_agent import GeminiAgent
from app.agents.mentor_agent import MentorAgent
# NOTE: GeminiAgent and MentorAgent are instantiated below

load_dotenv()

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- AGENT INITIALIZATION (Fixes TypeError by instantiating at root level) ---
# MentorAgent internally initializes the GeminiAgent and DataEnrichmentAgent.
mentor_agent = MentorAgent() 
# The GeminiAgent instance is kept separately only for its unique method (extract_profile)
gemini_agent = mentor_agent.gemini_agent 
# ----------------------------------------------------------------------------


class ResumeInput(BaseModel):
    # This model is not used for the multipart endpoint, but is kept for structure
    resume_text: str

# --- File Processing Utility ---

def extract_text_from_file(file_content: bytes, filename: str) -> str:
    """Extracts text content from PDF or DOCX bytes."""
    logger.info(f"Processing uploaded file: {filename}")
    if filename.endswith('.pdf'):
        try:
            # Use fitz (PyMuPDF) for PDF handling
            doc = fitz.open(stream=file_content, filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            return text
        except Exception as e:
            logger.error(f"PDF Extraction Failed: {e}")
            raise HTTPException(status_code=500, detail="Failed to extract text from PDF.")
    
    elif filename.endswith(('.docx', '.doc')):
        try:
            # Use python-docx
            doc = Document(io.BytesIO(file_content))
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text
        except Exception as e:
            logger.error(f"DOCX Extraction Failed: {e}")
            raise HTTPException(status_code=500, detail="Failed to extract text from DOCX.")
            
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type. Please upload PDF or DOCX.")


@app.get("/")
def read_root():
    return {"status": "AI Career Mentor Backend Running"}


@app.post("/analyze-profile")
async def analyze_profile(
    resume_text: Annotated[Optional[str], Form()] = None,
    file: Annotated[Optional[UploadFile], File()] = None,
):
    """
    Handles file uploads (PDF/DOCX) or text input, processes the resume, 
    and returns comprehensive profile analysis.
    """
    logger.info("Received request for profile analysis.")
    
    text_to_analyze = ""
    
    # 1. Determine Source of Resume Text
    if file:
        file_content = await file.read()
        text_to_analyze = extract_text_from_file(file_content, file.filename)
    elif resume_text:
        text_to_analyze = resume_text
    
    if not text_to_analyze.strip():
        raise HTTPException(status_code=400, detail="No resume content provided for analysis.")

    try:
        # STEP 2: AWAIT Profile Extraction (Core data + chart metrics)
        profile = await gemini_agent.extract_profile(text_to_analyze)
        
        # STEP 3: AWAIT Recommendation Generation (Compilation, Roadmap, Projects)
        recommendations = await mentor_agent.generate_recommendations(profile)
        
        return recommendations
    
    except Exception as e:
        logger.error(f"Critical error during profile analysis: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed due to a critical backend error: {e}"
        )
