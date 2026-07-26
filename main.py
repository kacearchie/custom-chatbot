import os
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI

app = FastAPI(title="Custom Chatbot API")

# Define the explicit domains allowed to talk to this backend
# Replace these URLs with your actual production frontend and dev origins
ALLOWED_ORIGINS = [
    "https://your-frontend-domain.com",       # Production domain
    "https://your-username.github.io",          # GitHub Pages (if applicable)
    "http://localhost:5500",                    # Local VS Code Live Server
    "http://127.0.0.1:5500",                   # Local IP testing
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Initialize OpenAI client using environment variable
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

class Message(BaseModel):
    role: str = Field(..., description="Role of the speaker: 'system', 'user', or 'assistant'")
    content: str = Field(..., description="Text content of the message")

class ChatRequest(BaseModel):
    messages: List[Message]

@app.get("/")
def health_check():
    """Health check endpoint used by uptime monitors to keep Render awake."""
    return {"status": "ok", "message": "FastAPI Chatbot Backend is running."}

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    """Processes incoming chat history and returns the gpt-4o-mini completion."""
    try:
        # System prompt setting the assistant persona
        system_instruction = {
            "role": "system", 
            "content": "You are a helpful, concise, and professional AI assistant. Respond using Markdown formatting where appropriate."
        }
        
        # Combine system prompt with incoming user history
        full_messages = [system_instruction] + [msg.model_dump() for msg in request.messages]

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=full_messages,
            temperature=0.7,
            max_tokens=500
        )

        reply_content = completion.choices[0].message.content
        return {"response": reply_content}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))