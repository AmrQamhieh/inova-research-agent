from fastapi import FastAPI, HTTPException
from app.schemas import QueryRequest, QueryResponse
from app.llm import ask_llm

app = FastAPI(title="Inova Research Agent API")


@app.get("/")
def root():
    return {"message": "Inova Research Agent API is running."}


@app.get("/health")
def health():
    return {"status": "OK!"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    try:
        answer = ask_llm(request.text)
        return QueryResponse(response=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))