import logging
import uuid

from fastapi import FastAPI, HTTPException, Request
from langchain_core.messages import AIMessage, HumanMessage

from app.agent.graph import agent_graph
from app.database import (
    DatabaseConfigurationError,
    SessionLocal,
    get_engine,
)
from app.models import Base, Conversation
from app.schemas import QueryRequest, QueryResponse
from app.llm import LLMRateLimitError, LLMTransientError
from app.logging_config import configure_logging, monotonic_ms

configure_logging()
logger = logging.getLogger("app.api")

app = FastAPI(title="Inova Research Agent API")

try:
    Base.metadata.create_all(bind=get_engine())
except DatabaseConfigurationError as exc:
    raise RuntimeError(f"Application startup failed: {exc}") from exc


@app.get("/")
def root():
    return {"message": "Inova Research Agent API is running."}


@app.get("/health")
def health():
    return {"status": "OK!"}


@app.middleware("http")
async def request_logging(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    start_ms = monotonic_ms()
    response = None
    try:
        response = await call_next(request)
        return response
    finally:
        elapsed_ms = monotonic_ms() - start_ms
        logger.info(
            "http_request",
            extra={
                "event": "http_request",
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": getattr(response, "status_code", None),
                "latency_ms": elapsed_ms,
            },
        )


def build_thread_messages(thread_id: str, latest_user_text: str):
    db = SessionLocal()
    try:
        history = (
            db.query(Conversation)
            .filter(Conversation.thread_id == thread_id)
            .order_by(Conversation.created_at.asc(), Conversation.id.asc())
            .all()
        )
    finally:
        db.close()

    messages = []
    for item in history:
        messages.append(HumanMessage(content=item.question))
        messages.append(AIMessage(content=item.response))

    messages.append(HumanMessage(content=latest_user_text))
    return messages


from app.database import get_database_url

@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    try:
        start_ms = monotonic_ms()
        messages = build_thread_messages(request.thread_id, request.text)

        result = agent_graph.invoke(
            {
                "messages": messages,
                "route": None,
                "search_results": None,
                "tools_invoked": [],
            }
        )

        assistant_response = result["messages"][-1].content
        tools_invoked = result.get("tools_invoked") or []

        db = SessionLocal()
        try:
            conversation = Conversation(
                thread_id=request.thread_id,
                question=request.text,
                response=assistant_response,
            )

            db.add(conversation)
            db.flush()
            db.commit()

            db.refresh(conversation)

        finally:
            db.close()

        elapsed_ms = monotonic_ms() - start_ms
        logger.info(
            "query_completed",
            extra={
                "event": "query_completed",
                "thread_id": request.thread_id,
                "tools_invoked": tools_invoked,
                "latency_ms": elapsed_ms,
            },
        )

        return QueryResponse(
            thread_id=request.thread_id,
            response=assistant_response)

    except (LLMRateLimitError, LLMTransientError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc



@app.get("/history/{thread_id}")
def get_history(thread_id: str):
    db = SessionLocal()
    try:
        history = (
            db.query(Conversation)
            .filter(Conversation.thread_id == thread_id)
            .order_by(Conversation.created_at.asc(), Conversation.id.asc())
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