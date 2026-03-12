from pydantic import BaseModel


class QueryRequest(BaseModel):
    thread_id: str
    text: str


class QueryResponse(BaseModel):
    thread_id: str
    response: str