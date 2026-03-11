from fastapi import FastAPI, HTTPException
from langchain_core.messages import HumanMessage

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
                "messages": [HumanMessage(content=request.text)],
                "route": None,
                "search_results": None,
            },
            config={"configurable": {"thread_id": request.thread_id}},
        )

        assistant_response = result["messages"][-1].content

        db = SessionLocal()
        try:
            conversation = Conversation(
                thread_id=request.thread_id,
                question=request.text,
                response=assistant_response,
            )
            db.add(conversation)
            db.commit()
        finally:
            db.close()

        return QueryResponse(response=assistant_response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history/{thread_id}")
def get_history(thread_id: str):
    db = SessionLocal()
    try:
        history = (
            db.query(Conversation)
            .filter(Conversation.thread_id == thread_id)
            .order_by(Conversation.created_at.asc())
            .all()
        )

        return {
            "thread_id": thread_id,
            "messages": [
                {
                    "id": item.id,
                    "question": item.question,
                    "response": item.response,
                    "created_at": item.created_at,
                }
                for item in history
            ],
        }
    finally:
        db.close()