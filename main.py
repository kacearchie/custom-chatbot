import os
from typing import List, Dict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI(title="Health Sciences AI Learning Hub")

# CORS setup - Update domain list as needed
ALLOWED_ORIGINS = [
    "https://kacearchie.github.io",  # Live GitHub Pages URL
    "http://127.0.0.1:5500",         # Local testing
    "http://localhost:5500"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client using environment variable
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """
You are an expert Health Sciences and Pharmacology AI Tutor assisting university students in medical, pharmaceutical, and healthcare disciplines.

### Guidelines:
1. Break down drugs and physiological processes systematically:
   - Class / Category
   - Mechanism of Action (MoA)
   - Therapeutic Uses / Indications
   - Key Adverse Effects & Contraindications
2. Break complex pathways (ADME, receptor signaling, inflammatory cascades) into numbered steps.
3. Keep explanations structured with bold headings, bullet points, and markdown.
4. Conclude major explanations with 1 short practice question to reinforce learning.
5. You are an academic tutor. Do not offer clinical diagnoses or personal medical advice.
"""

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    history: List[Message]

@app.get("/")
def health_check():
    return {"status": "healthy", "service": "Health Sciences AI API"}

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY environment variable missing.")

    try:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        # Append up to last 10 messages for conversation state
        for msg in request.history[-10:]:
            messages.append({"role": msg.role, "content": msg.content})

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7
        )

        reply = response.choices[0].message.content
        return {"reply": reply}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))