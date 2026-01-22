import os
import json
import httpx
import logging
import asyncio
from typing import Dict, Any

logger = logging.getLogger(__name__)

# --- Configuration Constants for Resilience ---
MAX_RETRIES = 5
INITIAL_BACKOFF = 5 # seconds

async def send_gemini_request_with_retry(
    api_key: str, 
    model: str, 
    body: Dict[str, Any], 
    purpose: str
) -> Dict:
    """
    Sends a request to the Gemini API with built-in Exponential Backoff 
    and Retry logic, primarily targeting 503 and transient network errors.
    """
    if not api_key:
        raise ValueError("API key is missing.")

    api_url = f"https://generativelanguage.googleapis.com/v1/models/{model}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    
    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"--- Sending {purpose} Request (Attempt {attempt + 1}) ---")
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(api_url, headers=headers, json=body)
                
                # Check for 503 Service Unavailable or 429 Rate Limit errors
                if response.status_code in [503, 429]:
                    if attempt < MAX_RETRIES - 1:
                        # Exponential backoff: 5s, 10s, 20s, 40s...
                        delay = INITIAL_BACKOFF * (2 ** attempt)
                        logger.warning(
                            f"Server returned {response.status_code}. Retrying in {delay}s...")
                        await asyncio.sleep(delay)
                        continue # Go to next attempt
                    else:
                        # If final attempt fails, raise the error
                        response.raise_for_status()

                # Handle all other HTTP errors (4xx, 5xx) immediately
                response.raise_for_status() 

                return response.json()

        except httpx.HTTPStatusError as e:
            # Catch all non-retryable HTTP errors (like 400, 403, 404)
            error_details = e.response.json().get("error", {})
            logger.error(
                f"HTTP Error {e.response.status_code} during {purpose}: {error_details.get('message', 'Unknown error')}")
            raise e
            
        except Exception as e:
            # Catch network errors, timeouts, etc.
            if attempt < MAX_RETRIES - 1:
                delay = INITIAL_BACKOFF * (2 ** attempt)
                logger.warning(f"Network error during {purpose}: {e}. Retrying in {delay}s...")
                await asyncio.sleep(delay)
                continue
            else:
                logger.error(f"Network/General Error during {purpose}: {e}")
                raise e
                
    # This line should be unreachable
    raise Exception(f"Failed to complete request for {purpose} after {MAX_RETRIES} attempts.")
