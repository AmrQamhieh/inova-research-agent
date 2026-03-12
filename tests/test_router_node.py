from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

from app.agent.nodes import router_node


def test_router_node_returns_general_route():
    state = {
        "messages": [
            HumanMessage(content="Hi"),
            AIMessage(content="Hello!"),
            HumanMessage(content="Explain what FastAPI is."),
        ],
        "route": None,
        "search_results": None,
    }

    with patch("app.agent.nodes.ask_llm", return_value="general"):
        out = router_node(state)

    assert out == {"route": "general"}


def test_router_node_returns_search_route_when_llm_says_search():
    state = {
        "messages": [HumanMessage(content="What's the stock price of NVDA right now?")],
        "route": None,
        "search_results": None,
    }

    with patch("app.agent.nodes.ask_llm", return_value="search"):
        out = router_node(state)

    assert out == {"route": "search"}


def test_router_node_is_strict_about_output_normalization():
    state = {
        "messages": [HumanMessage(content="Need latest news about X.")],
        "route": None,
        "search_results": None,
    }

    # Any output containing "search" should map to route="search"
    with patch("app.agent.nodes.ask_llm", return_value="SEARCH\n"):
        out = router_node(state)

    assert out["route"] == "search"
