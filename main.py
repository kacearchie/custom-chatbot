import os
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI(title="Custom Chatbot API with History")

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


# Structure for individual chat messages
class Message(BaseModel):
    role: str  # "system", "user", or "assistant"
    content: str


# Request body expecting full message history
class ChatRequest(BaseModel):
    messages: List[Message]


@app.get("/")
def health_check():
    return {"status": "ok", "message": "FastAPI Chatbot Backend is running."}


@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    if not request.messages:
        raise HTTPException(status_code=400, detail="Message history cannot be empty.")

    # Base system instructions
    system_instruction = {
        "role": "system",
        "content": "You are a helpful, friendly, and concise AI assistant integrated into a web widget.",
    }

    # Prepend system prompt to user conversation history
    full_conversation = [system_instruction] + [
        msg.model_dump() for msg in request.messages
    ]

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=full_conversation,
            temperature=0.7,
            max_tokens=500,
        )

        ai_response = completion.choices[0].message.content
        return {"response": ai_response}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing request: {str(e)}"
        )