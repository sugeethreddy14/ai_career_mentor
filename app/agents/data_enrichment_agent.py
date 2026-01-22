import os
import json
import logging
from typing import Dict, Any, List
# Import the shared service function
from app.services.api_service import send_gemini_request_with_retry 

logger = logging.getLogger(__name__)

class DataEnrichmentAgent:
    """
    DataEnrichmentAgent specializes in complex content generation (Roadmap and Projects)
    by running a targeted, high-temperature LLM call.
    """
    def __init__(self):
        self.api_key = os.getenv("LLM_API_KEY")
        self.model = "gemini-2.5-flash"
        
    async def generate_advanced_content(self, profile: Dict) -> Dict:
        """
        Generates detailed roadmap and advanced mini-projects based on the profile,
        using a high-quality prompt and large token budget.
        """
        if not self.api_key:
            return {"learning_roadmap": ["API Key missing."], "projects": ["API Key missing."]}

        skills_str = ", ".join(profile.get("skills", []))
        roles_str = ", ".join(profile.get("suggested_roles", ["AI Developer"]))
        
        # --- HIGH-QUALITY GENERATION PROMPT ---
        # The prompt is designed to force detailed, non-generic steps.
        prompt = (
            "Based on the user's primary skills ({skills_str}) and target roles ({roles_str}), "
            "provide a highly detailed, advanced learning roadmap (5 steps) and 5 advanced, specific mini-project ideas. "
            "The roadmap steps MUST be granular (e.g., 'Learn X architecture'), not generic (e.g., 'Learn advanced Python'). "
            "Format the output STRICTLY as the JSON object below. Do NOT include any surrounding text or markdown blocks.\n\n"
            
            f"Context: User is aiming for roles like {roles_str}.\n"
            f"Example Roadmap Step: 'Deep Dive into Generative Adversarial Networks (GANs) Architecture and PyTorch implementation.'\n"
            f"Example Project: 'Build a full-stack, real-time object detection pipeline using YOLOv8, Docker, and AWS Sagemaker.'\n\n"
            
            """
            {
              "learning_roadmap": ["<detailed step 1>", "<detailed step 2>", "<detailed step 3>", "<detailed step 4>", "<detailed step 5>"],
              "projects": ["<advanced project 1>", "<advanced project 2>", "<advanced project 3>", "<advanced project 4>", "<advanced project 5>"]
            }
            """
        )

        # NOTE: Using a massive token budget to prevent MAX_TOKENS error here.
        body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.6, # Higher temp for creative/detailed generation
                "maxOutputTokens": 5000, 
            }
        }
        
        try:
            # Use the shared service function for the API call
            result = await send_gemini_request_with_retry(
                api_key=self.api_key,
                model=self.model,
                body=body,
                purpose="Advanced Content Generation"
            )
            
            candidates = result.get("candidates", [])
            if not candidates: raise ValueError("Response lacks candidates list.")

            output_text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            if not output_text: raise ValueError("Model generated empty text content.")

            # Safely extract and parse JSON
            start = output_text.index("{")
            end = output_text.rindex("}") + 1
            advanced_data = json.loads(output_text[start:end])
            
            return advanced_data

        except Exception as e:
            logger.error(f"❌ Error generating advanced content: {e}. Returning fallback content.")
            return {"learning_roadmap": ["Failed to generate advanced LLM roadmap. Check logs."], "projects": ["Failed to generate advanced LLM projects. Check logs."]}
