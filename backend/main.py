from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any
from langgraph_.langgraph_main import text2sql

app = FastAPI()

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str

@app.post("/chating")
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    try:
        # text2sql 함수 호출
        result = text2sql(request.question)
        
        return ChatResponse(
            answer=result['answer']
        )
    
    except Exception as e:
        return ChatResponse(
            answer=f"죄송합니다. 오류가 발생했습니다: {str(e)}"
        )

# uvicorn main:app --reload로 실행