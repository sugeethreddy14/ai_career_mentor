import httpx
import os
from typing import List

class APIClient:
    def __init__(self):
        self.api_key = os.getenv("JOB_API_KEY")
        self.base_url = "https://jsearch.p.rapidapi.com/search"
        self.headers = {
            "X-RapidAPI-Key": self.api_key,
            "X-RapidAPI-Host": "jsearch.p.rapidapi.com"
        }

    def fetch_trending_roles(self, skills: List[str]) -> List[str]:
        query = " ".join(skills) if skills else "data"
        params = {"query": query, "num_pages": "1"}
        with httpx.Client(timeout=10.0) as client:
            response = client.get(self.base_url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()
            jobs = data.get("data", [])
            return [job.get("job_title", "") for job in jobs][:5]
