# import os
# import json
# import httpx
# import re
# from typing import Dict, Any, List
# from dotenv import load_dotenv
# import asyncio

# load_dotenv()

# class GeminiAgent:
#     """
#     Agent to extract and generate structured data from resume text using the Gemini API.
#     (Contains the internal helper functions and robust error handling.)
#     """
#     def __init__(self):
#         self.api_key = os.getenv("LLM_API_KEY")
#         if not self.api_key:
#             print("⚠️ Gemini API key not found; fallback extraction only.")
            
#         self.model = "gemini-2.5-flash" 
#         self.base_url = "https://generativelanguage.googleapis.com/v1" 
#         self.api_path = f"/models/{self.model}:generateContent"
#         self.api_url_template = f"{self.base_url}{self.api_path}"

#     async def _send_request(self, prompt: str, max_tokens: int = 512, temperature: float = 0.3) -> Dict:
#         """Helper function to send a configured API request."""
        
#         if not self.api_key:
#             raise Exception("API key is missing.")

#         headers = {"Content-Type": "application/json"}
#         body = {
#             "contents": [{"parts": [{"text": prompt}]}],
#             "generationConfig": {
#                 "temperature": temperature, 
#                 "maxOutputTokens": max_tokens, 
#             }
#         }
        
#         full_url = f"{self.api_url_template}?key={self.api_key}"

#         async with httpx.AsyncClient(timeout=120.0) as client:
#             response = await client.post(full_url, headers=headers, json=body)
#             response.raise_for_status() 
#             result = response.json()

#             candidates = result.get("candidates", [])
#             if not candidates:
#                  raise ValueError("Response lacks candidates list.")

#             candidate = candidates[0]
#             finish_reason = candidate.get("finishReason")
#             if finish_reason != "STOP":
#                 raise ValueError(f"Content stopped prematurely due to: {finish_reason}")

#             output_text = candidate.get("content", {}).get("parts", [{}])[0].get("text", "")
            
#             if not output_text:
#                 raise ValueError("Model generated empty text content.")
            
#             return {"text": output_text}


#     async def extract_profile(self, resume_text: str) -> Dict[str, Any]:
#         """Performs the initial structured data extraction."""
#         if not self.api_key:
#              return self.fallback_extract(resume_text, fallback_only=True)
             
#         # JSON template for the structured profile data
#         json_template = """
# {
#   "skills": ["<list of 10 primary technical skills>"],
#   "interests": ["<list of 5 key professional interests>"],
#   "education": ["<list of degrees and institutions>"],
#   "certifications": ["<list of certifications>"],
#   "level": "<Career level (Beginner, Intermediate, Advanced)>",
#   "suggested_roles": ["<5 highly relevant job titles>", "<job title 2>", "<job title 3>", "<job title 4>", "<job title 5>"]
# }
#         """

#         prompt = (
#             "Analyze the RESUME provided below and fill the JSON TEMPLATE exactly. "
#             "Your ONLY output must be the completed JSON object. "
#             "You MUST populate 'suggested_roles' with 5 job titles based on the extracted skills, NEVER an empty list.\n\n"
#             f"RESUME:\n\"\"\"{resume_text}\"\"\"\n\n"
#             f"JSON TEMPLATE TO FILL:\n{json_template}"
#         )

#         try:
#             print("\n--- Sending Extraction Request ---")
            
#             response = await self._send_request(prompt, max_tokens=2048, temperature=0.2)
#             output_text = response["text"]

#             try:
#                 start = output_text.index("{")
#                 end = output_text.rindex("}") + 1
#                 json_str = output_text[start:end]
#             except ValueError:
#                 json_str = output_text 

#             profile_data = json.loads(json_str)
#             return profile_data

#         except Exception as e:
#             print(f"❌ General Error during Profile Extraction: {e}. Using fallback extractor.")
#             return self.fallback_extract(resume_text, fallback_only=True)


#     async def generate_advanced_content(self, profile: Dict) -> Dict[str, List[str]]:
#         """
#         Generates detailed, high-quality learning roadmap and project ideas using the LLM.
#         """
        
#         suggested_roles = profile.get("suggested_roles", [])
#         primary_skills = profile.get("skills", [])
        
#         if not suggested_roles or not primary_skills:
#             return {"learning_roadmap": ["Cannot generate detailed roadmap without suggested roles/skills."], "projects": []}
            
#         roles_str = ", ".join(suggested_roles[:2])
#         skills_str = ", ".join(primary_skills[:5])
        
#         # FINAL CONTENT QUALITY PROMPT: With specific examples and constraints
#         prompt = (
#             f"The user aims for the roles: {roles_str}. Their current skills are: {skills_str}. "
#             "Your task is to provide an ADVANCED, detailed learning roadmap (5 steps) and 5 advanced, specific mini-project ideas. "
#             "Instructions:\n"
#             "1. **Roadmap:** Each step must be a complex, specific learning module focusing on theory/implementation gap closure.\n"
#             "2. **Projects:** Each project must be full-stack, end-to-end, or production-ready concepts.\n"
#             "3. **Output:** Return ONLY a JSON object with two keys: 'learning_roadmap' (List[str]) and 'projects' (List[str]).\n"
#             "\n"
#             "Example Roadmap Step: 'Mastering Attention Mechanisms in Transformer Models and implementing a custom training loop in PyTorch.'\n"
#             "Example Project: 'Build a personalized Diffusion Model for Image-to-Image Style Transfer deployed on a Dockerized FastAPI endpoint.'\n"
#         )

#         try:
#             print("\n--- Sending Advanced Content Request (Quality) ---")
            
#             # 💡 FIX: Increased max_tokens from 3000 to 5000 for maximum safety against the MAX_TOKENS error.
#             response = await self._send_request(prompt, max_tokens=5000, temperature=0.6) 
            
#             output_text = response["text"]
            
#             # Simple cleanup for JSON parsing
#             output_text = output_text.strip().replace("```json", "").replace("```", "")
            
#             content_data = json.loads(output_text)
            
#             return content_data
            
#         except Exception as e:
#             print(f"❌ Error generating advanced content: {e}. Returning fallback content.")
#             return {"learning_roadmap": ["Failed to generate advanced LLM roadmap."], "projects": ["Failed to generate advanced LLM projects."]}


#     def fallback_extract(self, text: str, fallback_only: bool = False) -> Dict[str, Any]:
#         """Provides a safe, basic fallback structure."""
#         technical_skills = re.findall(
#             r"\b(Python|Java|SQL|AWS|Docker|TensorFlow|PyTorch|NLP|Machine Learning|Deep Learning)\b",
#             text, re.I)
#         unique_skills = list(dict.fromkeys([skill.title() for skill in technical_skills]))[:10]

#         interests = []
#         interest_pattern = re.compile(r"(passion|interest|enthusiast|curious|love|enjoy)", re.I)
#         for sentence in re.split(r'[.?!\n]', text):
#             if interest_pattern.search(sentence):
#                 interests.append(sentence.strip())
#             if len(interests) >= 5:
#                 break
        
#         certifications = re.findall(
#             r"(AWS|Coursera|IBM|Stanford|DeepLearning\.AI|Google Cloud|Microsoft Certified)", text, re.I)
#         certifications = list(dict.fromkeys([cert.title() for cert in certifications]))

#         level = "Intermediate"
#         if re.search(r"\b(senior|expert|lead|principal|director)\b", text, re.I):
#             level = "Advanced"

#         suggested_roles = ["Data Scientist", "Machine Learning Engineer", "AI Specialist"]

#         data = {
#             "skills": unique_skills,
#             "interests": interests,
#             "education": ["(Fallback extraction cannot guarantee accuracy)"], 
#             "certifications": certifications,
#             "level": level,
#             "suggested_roles": suggested_roles 
#         }
        
#         if fallback_only:
#             data["status"] = "fallback_success"
            
#         return data

# import os
# import json
# import httpx
# import re
# from typing import Dict, Any, List
# from dotenv import load_dotenv
# import asyncio
# import logging

# load_dotenv()
# logger = logging.getLogger(__name__)

# # --- Configuration Constants ---
# MAX_RETRIES = 3
# INITIAL_BACKOFF = 2 # seconds

# class GeminiAgent:
#     """
#     Agent responsible for all direct communication with the Gemini API.
#     Handles core profile extraction and high-quality content generation 
#     (Roadmap/Projects) with built-in retry logic.
#     """
#     def __init__(self):
#         self.api_key = os.getenv("LLM_API_KEY")
#         self.model = "gemini-2.5-flash" 
#         self.base_url = "https://generativelanguage.googleapis.com/v1" 
#         self.api_url_template = f"{self.base_url}/models/{self.model}:generateContent"

#     async def _send_request_with_retry(self, body: Dict, purpose: str) -> Dict:
#         """Implements exponential backoff for resilience against 503 errors."""
#         full_url = f"{self.api_url_template}?key={self.api_key}"
#         headers = {"Content-Type": "application/json"}
        
#         for attempt in range(MAX_RETRIES):
#             try:
#                 logger.info(f"--- Sending {purpose} Request (Attempt {attempt + 1}) ---")
#                 async with httpx.AsyncClient(timeout=120.0) as client:
#                     response = await client.post(full_url, headers=headers, json=body)
                    
#                     # Handle 503 Service Unavailable with retry logic
#                     if response.status_code == 503:
#                         if attempt < MAX_RETRIES - 1:
#                             delay = INITIAL_BACKOFF * (2 ** attempt)
#                             logger.warning(f"Server returned 503. Retrying in {delay}s...")
#                             await asyncio.sleep(delay)
#                             continue # Go to next attempt
#                         else:
#                             # If final attempt fails, raise for handling
#                             response.raise_for_status() 

#                     # Handle all other HTTP errors (4xx, 5xx) immediately
#                     response.raise_for_status() 

#                     return response.json()

#             except httpx.HTTPStatusError as e:
#                 # Catch and log the specific error
#                 error_details = e.response.json().get("error", {})
#                 logger.error(f"HTTP Error {e.response.status_code} during {purpose}: {error_details.get('message', 'Unknown error')}")
#                 raise e
#             except Exception as e:
#                 # Catch network errors, etc.
#                 logger.error(f"Network/General Error during {purpose}: {e}")
#                 raise e
                
#         # This line should technically be unreachable if MAX_RETRIES > 0
#         raise Exception(f"Failed to complete request for {purpose} after {MAX_RETRIES} attempts.")

#     async def extract_profile(self, resume_text: str) -> Dict[str, Any]:
#         """Extracts core profile data and chart metrics."""
#         if not self.api_key:
#             return self.fallback_extract(resume_text, fallback_only=True)

#         prompt = (
#             "Analyze the RESUME provided below and fill the JSON TEMPLATE exactly. "
#             "You MUST score the top 5 technical skills on a scale of 1 to 10 based on evidence (experience, certifications, detail) in the resume. "
#             "You MUST provide 5 relevant job titles for 'suggested_roles'. "
#             "Return ONLY the completed JSON object, with NO surrounding text or markdown blocks.\n\n"
            
#             # --- STRUCTURED TEMPLATE (for core data + charts) ---
#             """
#             {
#               "skills": ["<list of 10 primary technical skills>"],
#               "interests": ["<list of 5 key professional interests>"],
#               "education": ["<list of degrees and institutions>"],
#               "certifications": ["<list of certifications>"],
#               "level": "<Career level (Beginner, Intermediate, Advanced)>",
#               "suggested_roles": ["<5 highly relevant job titles>", "..."],
#               "skill_scores": {"<Skill 1>": <score 1-10>, "...": "..."},
#               "critical_knowledge_gaps": ["<List of 3 crucial missing topics/skills>"]
#             }
#             """
#             f"\n\nRESUME:\n\"\"\"{resume_text}\"\"\""
#         )
        
#         # NOTE: This uses the highest complexity prompt, but its failure is handled by the retry/fallback logic.
#         body = {
#             "contents": [{"parts": [{"text": prompt}]}],
#             "generationConfig": {
#                 "temperature": 0.3, 
#                 "maxOutputTokens": 2048, # High enough for extraction
#             }
#         }

#         try:
#             result = await self._send_request_with_retry(body, "Core Profile Extraction")
            
#             # ... (rest of the extraction and parsing logic remains the same)
            
#             candidates = result.get("candidates", [])
#             if not candidates: raise ValueError("Response lacks candidates list.")

#             candidate = candidates[0]
#             if candidate.get("finishReason") != "STOP":
#                 raise ValueError(f"Content stopped prematurely: {candidate.get('finishReason')}")

#             output_text = candidate.get("content", {}).get("parts", [{}])[0].get("text", "")
#             if not output_text: raise ValueError("Model generated empty text content.")

#             # Safely extract and parse JSON
#             start = output_text.index("{")
#             end = output_text.rindex("}") + 1
#             profile_data = json.loads(output_text[start:end])
            
#             return profile_data

#         except Exception as e:
#             logger.error(f"❌ General Error during Profile Extraction: {e}. Using fallback extractor.")
#             return self.fallback_extract(resume_text, fallback_only=True)

#     async def generate_advanced_content(self, profile: Dict) -> Dict:
#         """Generates detailed roadmap and projects based on the extracted profile."""
#         if not self.api_key:
#             return {"learning_roadmap": ["API Key missing."], "projects": ["API Key missing."]}

#         skills_str = ", ".join(profile.get("skills", []))
#         roles_str = ", ".join(profile.get("suggested_roles", ["AI Developer"]))
        
#         # --- HIGH-QUALITY GENERATION PROMPT ---
#         prompt = (
#             "Based on the user's primary skills ({skills_str}) and target roles ({roles_str}), "
#             "provide a highly detailed, advanced learning roadmap (5 steps) and 5 advanced, specific mini-project ideas. "
#             "Format the output STRICTLY as the JSON object below. Do NOT include any surrounding text or markdown blocks.\n\n"
            
#             # Using 3,000 token output to guarantee space for detail
#             f"Example Roadmap Step: 'Deep Dive into Generative Adversarial Networks (GANs) Architecture (e.g., CycleGANs, StyleGANs).' "
#             f"Example Project: 'Build a full-stack, real-time object detection pipeline using YOLOv8, Docker, and AWS Sagemaker.'\n\n"
            
#             """
#             {
#               "learning_roadmap": ["<detailed step 1>", "<detailed step 2>", "<detailed step 3>", "<detailed step 4>", "<detailed step 5>"],
#               "projects": ["<advanced project 1>", "<advanced project 2>", "<advanced project 3>", "<advanced project 4>", "<advanced project 5>"]
#             }
#             """
#         )

#         # NOTE: Using a massive token budget to prevent MAX_TOKENS error here.
#         body = {
#             "contents": [{"parts": [{"text": prompt}]}],
#             "generationConfig": {
#                 "temperature": 0.6, # Higher temp for creative/detailed generation
#                 "maxOutputTokens": 5000, 
#             }
#         }
        
#         try:
#             result = await self._send_request_with_retry(body, "Advanced Content Generation")
            
#             candidates = result.get("candidates", [])
#             if not candidates: raise ValueError("Response lacks candidates list.")

#             output_text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
#             if not output_text: raise ValueError("Model generated empty text content.")

#             # Safely extract and parse JSON
#             start = output_text.index("{")
#             end = output_text.rindex("}") + 1
#             advanced_data = json.loads(output_text[start:end])
            
#             return advanced_data

#         except Exception as e:
#             logger.error(f"❌ Error generating advanced content: {e}. Returning fallback content.")
#             return {"learning_roadmap": ["Failed to generate advanced LLM roadmap. Check logs."], "projects": ["Failed to generate advanced LLM projects. Check logs."]}

#     def fallback_extract(self, text: str, fallback_only: bool = False) -> Dict[str, Any]:
#         """Simple regex-based extraction used when the API call fails."""
#         # --- Fallback logic remains unchanged ---
#         technical_skills = re.findall(
#             r"\b(Python|Java|SQL|AWS|Docker|TensorFlow|PyTorch|NLP|Machine Learning|Deep Learning)\b",
#             text, re.I)
#         unique_skills = list(dict.fromkeys([skill.title() for skill in technical_skills]))[:10]

#         suggested_roles = ["Data Scientist", "Machine Learning Engineer", "AI Specialist"]

#         data = {
#             "skills": unique_skills,
#             "interests": [],
#             "education": ["(Fallback extraction cannot guarantee accuracy)"], 
#             "certifications": [],
#             "level": "Intermediate",
#             "suggested_roles": suggested_roles,
#             "skill_scores": {},
#             "critical_knowledge_gaps": ["Fallback mode active. Check API key/connection."]
#         }
        
#         if fallback_only:
#             data["status"] = "fallback_success"
            
#         return data


import os
import json
import re
from typing import Dict, Any, List
from dotenv import load_dotenv
import logging
import asyncio # <-- FIX: Ensure asyncio is explicitly imported
# Import the shared service function
from app.services.api_service import send_gemini_request_with_retry 

load_dotenv()
logger = logging.getLogger(__name__)

class GeminiAgent:
    """
    GeminiAgent is responsible for the core extraction of structured data from the resume.
    It uses the shared API service for robust communication with the Gemini API.
    """
    def __init__(self):
        self.api_key = os.getenv("LLM_API_KEY")
        self.model = "gemini-2.5-flash" 

    async def extract_profile(self, resume_text: str) -> Dict[str, Any]:
        """Extracts core profile data and chart metrics via a single LLM call."""
        if not self.api_key:
            return self.fallback_extract(resume_text, fallback_only=True)

        prompt = (
            "Analyze the RESUME provided below and fill the JSON TEMPLATE exactly. "
            "You MUST score the top 5 technical skills on a scale of 1 to 10 based on evidence (experience, certifications, detail) in the resume. "
            "You MUST provide 5 relevant job titles for 'suggested_roles'. "
            "Return ONLY the completed JSON object, with NO surrounding text or markdown blocks.\n\n"
            
            # --- STRUCTURED TEMPLATE (for core data + charts) ---
            """
            {
              "skills": ["<list of 10 primary technical skills>"],
              "interests": ["<list of 5 key professional interests>"],
              "education": ["<list of degrees and institutions>"],
              "certifications": ["<list of certifications>"],
              "level": "<Career level (Beginner, Intermediate, Advanced)>",
              "suggested_roles": ["<5 highly relevant job titles>", "..."],
              "skill_scores": {"<Top Skill 1>": <score 1-10>, "...": "..."},
              "critical_knowledge_gaps": ["<List of 3 crucial missing topics/skills>"]
            }
            """
            f"\n\nRESUME:\n\"\"\"{resume_text}\"\"\""
        )
        
        body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.3, 
                # FIX: Increased capacity to handle complex structured output
                "maxOutputTokens": 4096, 
            }
        }

        try:
            # Use the shared service function for the API call
            result = await send_gemini_request_with_retry(
                api_key=self.api_key,
                model=self.model,
                body=body,
                purpose="Core Profile Extraction"
            )
            
            candidates = result.get("candidates", [])
            if not candidates: raise ValueError("Response lacks candidates list.")

            candidate = candidates[0]
            if candidate.get("finishReason") != "STOP":
                raise ValueError(f"Content stopped prematurely: {candidate.get('finishReason')}")

            output_text = candidate.get("content", {}).get("parts", [{}])[0].get("text", "")
            if not output_text: raise ValueError("Model generated empty text content.")

            # Safely extract and parse JSON
            start = output_text.index("{")
            end = output_text.rindex("}") + 1
            profile_data = json.loads(output_text[start:end])
            
            return profile_data

        except Exception as e:
            logger.error(f"❌ General Error during Profile Extraction: {e}. Using fallback extractor.")
            # Ensure the fallback structure is returned if the API call fails
            return self.fallback_extract(resume_text, fallback_only=True)

    def fallback_extract(self, text: str, fallback_only: bool = False) -> Dict[str, Any]:
        """Simple regex-based extraction used when the API call fails."""
        technical_skills = re.findall(
            r"\b(Python|Java|SQL|AWS|Docker|TensorFlow|PyTorch|NLP|Machine Learning|Deep Learning)\b",
            text, re.I)
        unique_skills = list(dict.fromkeys([skill.title() for skill in technical_skills]))[:10]

        suggested_roles = ["Data Scientist", "Machine Learning Engineer", "AI Specialist"]

        data = {
            "skills": unique_skills,
            "interests": [],
            "education": ["(Fallback extraction cannot guarantee accuracy)"], 
            "certifications": [],
            "level": "Intermediate",
            "suggested_roles": suggested_roles,
            "skill_scores": {},
            "critical_knowledge_gaps": ["Fallback mode active. Check API key/connection."]
        }
        
        if fallback_only:
            data["status"] = "fallback_success"
            
        return data


# =============================================================================
# Example usage (Test Runner)
# ==============================================================================
if __name__ == "__main__":
    
    sample_resume = """
    Tapasvi Panchagnula Hyderabad, Telangana — +91 9063702246 — Tapasvi5fires@gmail.com https://www.linkedin.com/in/tapasvi-panchagnula-96986227b/ — https://github.com/Tapasvi5fires Objective AI & ML enthusiast with strong passion and expertise in Data Science, Deep Learning, and Generative AI. Experienced in building intelligent systems, conducting applied research, and delivering AI solutions to real-world problems. Seeking opportunities as a Machine Learning Engineer, Data Scientist, or Data Anal...
    """
    
    print("--- Starting Resume Extraction ---")
    
    agent = GeminiAgent()
    profile = asyncio.run(agent.extract_profile(sample_resume))
    
    print("\n--- Extracted Profile ---")
    print(json.dumps(profile, indent=2))
    print("-------------------------")
