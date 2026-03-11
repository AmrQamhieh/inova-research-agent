from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from app.agent.state import AgentState
from app.agent.nodes import (
    router_node,
    general_node,
    search_node,
    answer_with_search_node,
)

memory = MemorySaver()


def route_decision(state: AgentState) -> str:
    return state["route"]


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("router", router_node)
    graph.add_node("general", general_node)
    graph.add_node("search", search_node)
    graph.add_node("answer_with_search", answer_with_search_node)

    graph.add_edge(START, "router")

    graph.add_conditional_edges(
        "router",
        route_decision,
        {
            "general": "general",
            "search": "search",
        },
    )

    graph.add_edge("general", END)
    graph.add_edge("search", "answer_with_search")
    graph.add_edge("answer_with_search", END)

    return graph.compile(checkpointer=memory)


agent_graph = build_graph()