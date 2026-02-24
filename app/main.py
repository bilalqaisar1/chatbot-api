from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Get API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not found in environment variables")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize FastAPI app
app = FastAPI(
    title="FastAPI OpenAI Chatbot",
    description="Chatbot API using FastAPI and OpenAI",
    version="1.0.0"
)

# ======================
# Request / Response Models
# ======================

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    response: str

# ======================
# Health Check Endpoint
# ======================

@app.get("/")
def root():
    return {"status": "Chatbot API is running"}

# ======================
# Chatbot Endpoint
# ======================

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        result = client.responses.create(
            model="gpt-4.1-mini",
            input=request.query
        )

        reply = result.output_text

        return {"response": reply}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
