from pydantic import BaseModel


class QueryRequest(BaseModel):
    thread_id: str
    text: str


class QueryResponse(BaseModel):
    response: str