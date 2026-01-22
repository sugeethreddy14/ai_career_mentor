import os
import json
from typing import Dict, Any
from dotenv import load_dotenv
import asyncio
import re
from google import genai
from google.genai import types

# Load environment variables from a .env file (for LLM_API_KEY)
load_dotenv()

class GeminiAgentSDK:
    """
    Agent to extract structured data using the official Google GenAI SDK.
    """
    def __init__(self):
        self.api_key = os.getenv("LLM_API_KEY")
        if not self.api_key:
            print("⚠️ Gemini API key not found; API calls will fail.")
            
        self.model = "gemini-2.5-flash" 
        
        try:
            self.client = genai.Client(api_key=self.api_key)
        except Exception as e:
            print(f"Failed to initialize Gemini Client: {e}")
            self.client = None

    async def extract_profile(self, resume_text: str) -> Dict[str, Any]:
        if not self.client or not self.api_key:
            print("Gemini client not initialized; using fallback extractor.")
            return self.fallback_extract(resume_text)

        prompt = (
            "You are an expert career mentor AI assistant. Analyze the resume text below and extract:\n"
            "- top 10 technical skills\n- 5 key interests\n- Education & certifications\n"
            "- Career level (Beginner, Intermediate, Advanced)\n- Suggested roles aligned to the profile\n"
            "Return ONLY a valid JSON object with keys: skills (list of str), interests (list of str), "
            "education (list of str), certifications (list of str), level (str), suggested_roles (list of str)."
            "Do not include any text, notes, or markdown formatting outside of the single JSON object.\n\n"
            f"Resume:\n\"\"\"{resume_text}\"\"\""
        )
        
        # 💡 FIX APPLIED HERE: max_output_tokens is now 2048 (from 1024) to ensure completion.
        config = types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=2048,
        )

        try:
            print(f"\n🚀 Sending request to {self.model} via SDK...")
            
            response = self.client.models.generate_content(
                model=self.model,
                contents=[prompt],
                config=config,
            )

            # Check for blocking reason
            if response.candidates and response.candidates[0].finish_reason != types.FinishReason.STOP:
                print("\n--- Model Response Blocked/Stopped ---")
                print(f"Finish Reason: {response.candidates[0].finish_reason.name}")
                print(f"Prompt Feedback: {response.prompt_feedback}")
                print("--------------------------------------")
                raise ValueError(f"Content stopped prematurely due to: {response.candidates[0].finish_reason.name}")

            output_text = response.text

            # Safely extract JSON substring
            try:
                start = output_text.index("{")
                end = output_text.rindex("}") + 1
                json_str = output_text[start:end]
            except ValueError:
                print(f"⚠️ Warning: Model output not clean JSON. Attempting direct parse.")
                json_str = output_text 

            profile_data = json.loads(json_str)
            return profile_data

        except (ValueError, json.JSONDecodeError) as e:
            print(f"\n❌ Content/JSON Error: {e}")
            print("Using fallback extractor.")
            return self.fallback_extract(resume_text)
            
        except Exception as e:
            print(f"\n❌ SDK API Error: {e}")
            return self.fallback_extract(resume_text)

    def fallback_extract(self, text: str) -> Dict[str, Any]:
        """Simple regex-based extraction used when the API call fails."""
        technical_skills = re.findall(
            r"\b(Python|SQL|AWS|Docker|Machine Learning|NLP)\b", text, re.I)
        unique_skills = list(dict.fromkeys([skill.title() for skill in technical_skills]))[:10]
        
        level = "Intermediate" 
        if re.search(r"\b(senior|expert|lead|principal)\b", text, re.I):
            level = "Advanced"

        return {"status": "fallback_success", "level": level, "skills": unique_skills}

# ==============================================================================
# Example usage (Test Runner)
# ==============================================================================
if __name__ == "__main__":
    
    agent = GeminiAgentSDK() 

    sample_resume = """
    John Smith - Data Scientist
    Summary: A highly proficient Data Scientist with 5 years of experience specializing in Machine Learning, 
    Deep Learning, and large-scale data analysis. Passionate about MLOps and cloud deployment.
    Skills: Python (PyTorch, TensorFlow, Pandas), SQL, AWS, Docker, Kubernetes, NLP.
    Education: M.S. in Computer Science, State University (2018)
    Certification: AWS Certified Machine Learning - Specialty.
    """
    
    print("🚀 Starting Gemini Agent Test (Using SDK)...")
    
    profile = asyncio.run(agent.extract_profile(sample_resume))
    
    print("\n✅ Final Extracted Profile:")
    print(json.dumps(profile, indent=2))
    print("\nTest Complete.")