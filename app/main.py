import logging
import uuid

from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi import FastAPI, HTTPException, Request, Response
from langchain_core.messages import AIMessage, HumanMessage

from app.agent.graph import agent_graph
from app.database import (
    DatabaseConfigurationError,
    SessionLocal,
    get_engine,
)
from app.metrics import record_http_request, record_http_server_error, record_error
from app.models import Base, Conversation
from app.schemas import QueryRequest, QueryResponse
from app.llm import LLMRateLimitError, LLMTransientError
from app.logging_config import configure_logging, monotonic_ms

configure_logging()
logger = logging.getLogger("app.api")

app = FastAPI(title="Inova Research Agent API")

@app.get("/metrics", include_in_schema=False)
def metrics():
    # generate_latest() returns bytes. FastAPI's Response accepts bytes or str.
    # To be perfectly safe across all ASGI implementations, we return bytes.
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

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

    try:
        response = await call_next(request)
        elapsed_ms = monotonic_ms() - start_ms
        duration_seconds = elapsed_ms / 1000

        record_http_request(
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration=duration_seconds,
        )
        record_http_server_error(
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
        )

        logger.info(
            "http_request",
            extra={
                "event": "http_request",
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "latency_ms": elapsed_ms,
            },
        )

        return response

    except Exception as exc:
        elapsed_ms = monotonic_ms() - start_ms
        duration_seconds = elapsed_ms / 1000

        record_http_request(
            method=request.method,
            path=request.url.path,
            status_code=500,
            duration=duration_seconds,
        )
        record_http_server_error(
            method=request.method,
            path=request.url.path,
            status_code=500,
        )
        record_error(component="http", error_type=type(exc).__name__)

        logger.info(
            "http_request",
            extra={
                "event": "http_request",
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": 500,
                "latency_ms": elapsed_ms,
            },
        )

        raise


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
            response=assistant_response,
        )

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