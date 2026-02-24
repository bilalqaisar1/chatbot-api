from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not found")

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(
    title="FastAPI OpenAI Chatbot",
    description="Chatbot API using FastAPI and OpenAI",
    version="1.0.0"
)

# ✅ ADD THIS BLOCK
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    response: str

@app.get("/")
def root():
    return {"status": "Chatbot API is running"}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        result = client.responses.create(
            model="gpt-4.1-mini",
            input=request.query
        )
        return {"response": result.output_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))