from typing import Annotated, Optional, TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    route: Optional[str]
    search_results: Optional[str]