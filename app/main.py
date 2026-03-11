from fastapi import FastAPI, HTTPException
from app.schemas import QueryRequest, QueryResponse
from app.agent.graph import agent_graph
from app.database import engine, SessionLocal
from app.models import Base, Conversation

app = FastAPI(title="Inova Research Agent API")

Base.metadata.create_all(bind=engine)


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
        result = agent_graph.invoke(
            {
                "question": request.text,
                "route": None,
                "search_results": None,
                "response": None,
            }
        )

        db = SessionLocal()
        try:
            conversation = Conversation(
                question=request.text,
                response=result["response"],
            )
            db.add(conversation)
            db.commit()
        finally:
            db.close()

        return QueryResponse(response=result["response"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))