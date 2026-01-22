import os
from typing import Dict, List
import numpy as np
import logging
from sentence_transformers import SentenceTransformer
import asyncio
from app.services.api_client import APIClient
from app.services.rag_service import RAGService

# Import the necessary agents (dependencies)
from app.agents.llm_profile_agent import GeminiAgent
from app.agents.data_enrichment_agent import DataEnrichmentAgent

logger = logging.getLogger(__name__)

class MentorAgent:
    """
    MentorAgent orchestrates the entire analysis. It compiles data extracted by 
    the GeminiAgent, enriched by the DataEnrichmentAgent, and supplemented by RAG.
    """
    def __init__(self):
        # Initialize external services
        self.api_client = APIClient()
        vector_db_path = os.getenv("VECTOR_DB_PATH", "./data/skills_index.faiss")
        self.rag_service = RAGService(vector_db_path) 
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize LLM Agents internally (resolves the TypeError in main.py)
        self.gemini_agent = GeminiAgent() 
        self.enrichment_agent = DataEnrichmentAgent() 


    def embed_skills(self, skills: List[str]) -> np.ndarray:
        # NOTE: Using 384 dimensions as it's common for all-MiniLM-L6-v2
        d = 384 
        if not skills:
            return np.zeros((1, d), dtype='float32')
        try:
            embeddings = self.embedder.encode(skills)
            avg_emb = np.mean(embeddings, axis=0)
            return avg_emb.reshape(1, -1).astype('float32')
        except Exception as e:
            logger.error(f"Error during skill embedding: {e}")
            return np.zeros((1, d), dtype='float32')


    def clean_skill_query(self, skills: List[str]) -> str:
        # Cleans skills for external API query (e.g., job search API)
        safe_skills = [s for s in skills if s.replace(" ", "").isalpha()]
        return " ".join(safe_skills) if safe_skills else "data science"


    def determine_suggested_roles(self, profile: Dict) -> List[str]:
        """
        Ensures suggested roles are never empty by using fallbacks if the LLM fails 
        or returns a placeholder.
        """
        initial_roles = profile.get("suggested_roles", [])
        skills = profile.get("skills", [])
        
        # 1. Use Roles from the GeminiAgent if they look like real roles
        if initial_roles and all(len(role) > 5 for role in initial_roles):
            logger.info("Using roles provided by Gemini Agent.")
            return initial_roles
        
        # 2. Try External API for Trending Roles (Placeholder structure)
        query = self.clean_skill_query(skills)
        trending_roles = []
        try:
            trending_roles = self.api_client.fetch_trending_roles(query) 
        except Exception as e:
            logger.warning(f"Failed to fetch trending roles from API: {e}")

        if trending_roles:
            logger.info("Using roles from external API.")
            return trending_roles[:5]
        
        # 3. Internal Derivation Fallback (GUARANTEED OUTPUT)
        logger.warning("All external sources failed. Falling back to internal role derivation.")
        
        if "Generative AI" in skills or "NLP" in skills:
            derived_roles = ["Generative AI Engineer", "NLP Scientist", "Machine Learning Engineer", "AI Researcher"]
        elif "AWS" in skills and "Docker" in skills:
            derived_roles = ["MLOps Engineer", "Cloud Solutions Architect", "DevOps Engineer"]
        elif "SQL" in skills:
             derived_roles = ["Data Analyst", "Data Scientist", "Business Intelligence Specialist"]
        else:
            derived_roles = ["Technology Consultant", "Software Developer", "AI Specialist"]

        return derived_roles[:5]


    async def generate_recommendations(self, profile: Dict) -> Dict:
        """
        Orchestrates the entire recommendation process, calling sub-agents for data.
        """
        skills = profile.get("skills", [])
        
        # 1. Determine roles (safely)
        final_suggested_roles = self.determine_suggested_roles(profile)
        
        # 2. Add final roles back to the profile before sending to enrichment agent
        profile['suggested_roles'] = final_suggested_roles 
        
        # 3. ASYNC CALL: Generate high-quality roadmap and projects
        # This call is now correctly routed through the self.enrichment_agent instance.
        advanced_content_task = self.enrichment_agent.generate_advanced_content(profile)
        
        # 4. RAG: Embed profile for Similar Profiles (This is synchronous)
        user_embedding = self.embed_skills(skills)
        similar_profiles = self.rag_service.get_similar_profiles(user_embedding)
        
        # 5. Wait for the LLM enrichment to complete
        advanced_content = await advanced_content_task

        # 6. Compile the final structure
        return {
            "skills": skills,
            "interests": profile.get("interests", []),
            "level": profile.get("level", "Unknown"),
            
            "suggested_roles": final_suggested_roles,
            "similar_profiles": similar_profiles, 
            
            "learning_roadmap": advanced_content.get("learning_roadmap", []),
            "projects": advanced_content.get("projects", []),
            
            # New Chart Data Metrics
            "skill_scores": profile.get("skill_scores", {}), 
            "critical_knowledge_gaps": profile.get("critical_knowledge_gaps", []),
            
            # Pass-through fields
            "education": profile.get("education", []),
            "certifications": profile.get("certifications", []),
        }
