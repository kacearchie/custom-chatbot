import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

# Initialize FastAPI application
app = FastAPI(title="Custom Chatbot API")

# Configure CORS Middleware
# Allows frontend scripts from any origin (local files, localhost, live domains)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client using the environment variable OPENAI_API_KEY
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


# Define the request body structure
class ChatRequest(BaseModel):
    message: str


@app.get("/")
def health_check():
    """Simple health check endpoint to verify backend is up."""
    return {"status": "ok", "message": "FastAPI Chatbot Backend is running."}


@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    """Chat endpoint that forwards user messages to OpenAI and returns the AI's response."""
    user_message = request.message.strip()

    if not user_message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    try:
        # Call OpenAI Chat Completion API
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful, friendly, and concise AI assistant integrated into a web widget.",
                },
                {"role": "user", "content": user_message},
            ],
            temperature=0.7,
            max_tokens=500,
        )

        ai_response = completion.choices[0].message.content
        return {"response": ai_response}

    except Exception as e:
        # Handle API errors or credential issues gracefully
        raise HTTPException(
            status_code=500, detail=f"An error occurred while contacting OpenAI: {str(e)}"
        )