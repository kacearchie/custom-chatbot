import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List
from openai import OpenAI

app = FastAPI(title="Custom Chatbot API")

# Allow web browsers from any domain to connect during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to your domain in production (e.g., ["https://yourwebsite.com"])
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client using the environment variable set in Step 1
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

@app.get("/")
def read_root():
    return {"status": "Chatbot backend is running!"}

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    if not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY environment variable is missing.")

    try:
        # Request a streaming response from gpt-4o-mini
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": m.role, "content": m.content} for m in request.messages],
            stream=True
        )

        def event_stream():
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    # Format data as Server-Sent Event standard
                    yield f"data: {json.dumps({'content': content})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))