import os
from dotenv import load_dotenv

def load_api_key():
    
    load_dotenv()

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in .env file. Please add it.")
        
    return api_key

