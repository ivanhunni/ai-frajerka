# api/app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from inference.engine import InferenceEngine

app = FastAPI(title="AI-Frajerka API", version="1.0")

# Globálna načítaná inštancia motora
engine = None

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    system_prompt: Optional[str] = "Si milá, starostlivá a občas vtipná AI priateľka. Odpovedáš prirodzene v slovenskom jazyku."
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9

class ChatResponse(BaseModel):
    reply: str

@app.on_event("startup")
def startup_event():
    global engine
    engine = InferenceEngine(
        model_path="checkpoints/aligned_model/final_aligned_model.pt",
        tokenizer_path="tokenizer/tokenizer.json"
    )

@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if not engine:
        raise HTTPException(status_code=500, detail="Inference engine is not initialized.")
    
    try:
        dict_messages = [{"role": m.role, "content": m.content} for m in request.messages]
        reply_text = engine.generate_reply(
            messages=dict_messages,
            system_prompt=request.system_prompt,
            temperature=request.temperature,
            top_p=request.top_p
        )
        return ChatResponse(reply=reply_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)