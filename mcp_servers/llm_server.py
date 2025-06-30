import os
from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel

# --- Pydantic Models for Request/Response ---
class InvokeRequest(BaseModel):
    prompt: str

class InvokeResponse(BaseModel):
    response: str

# --- FastAPI App Initialization ---
app = FastAPI()

# --- Initialize OpenAI Client ---
# The client automatically reads the LLM_API_KEY and LLM_API_BASE from environment variables
try:
    client = OpenAI(
        api_key=os.environ.get("LLM_API_KEY"),
        base_url=os.environ.get("LLM_API_BASE"),
    )
    LLM_MODEL = os.environ.get("LLM_MODEL_NAME", "gpt-4o-mini")
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    client = None

# --- API Endpoints ---
@app.post("/invoke", response_model=InvokeResponse)
async def invoke_llm(request: InvokeRequest):
    if not client:
        raise HTTPException(status_code=500, detail="LLM client not initialized. Check API key and base URL.")
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": request.prompt,
                }
            ],
            model=LLM_MODEL,
        )
        response_content = chat_completion.choices[0].message.content
        return InvokeResponse(response=response_content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error communicating with LLM API: {e}")

@app.get("/healthz")
async def health_check():
    # For a public API, we can just check if the client is initialized
    if client:
        return {"status": "ok"}
    else:
        raise HTTPException(status_code=503, detail="Service Unavailable: LLM client not configured")