from app.agent.state import AgentState
from app.llm import ask_llm
from app.agent.tools import search_web
from app.metrics import record_tool_invocation, record_error
from langchain_core.messages import AIMessage, HumanMessage
import logging

logger = logging.getLogger("app.agent")


def router_node(state: AgentState) -> AgentState:
    question = get_latest_user_message(state)
    history = format_conversation_history(state)

    prompt = f"""
You are a routing assistant.

Decide whether the user's latest message can be answered from the existing conversation context
or general knowledge, or if it truly requires fresh/external information from the web.

Use these rules:
- Return "general" if the answer can be inferred from the conversation history or normal knowledge.
- Return "search" only if the user is asking for fresh, live, recent, or external factual information.

Return only one word:
- general
- search

Conversation History:
{history}

Latest User Question:
{question}
""".strip()

    decision = ask_llm(
        prompt,
        system_prompt="You are a strict classifier. Return only 'general' or 'search'."
    ).strip().lower()

    if "search" in decision:
        route = "search"
    else:
        route = "general"

    logger.info(
        "router_decision",
        extra={"event": "router_decision", "route": route},
    )

    return {
        "route": route
    }


def general_node(state: AgentState) -> AgentState:
    logger.info("general_node", extra={"event": "general_node"})

    question = get_latest_user_message(state)
    history = format_conversation_history(state)

    prompt = f"""
You are a helpful assistant.

Use the conversation history to answer the user's latest question.
If the answer is already available from the conversation context, use it directly.
Be concise and accurate.

Conversation History:
{history}

Latest User Question:
{question}
""".strip()

    answer = ask_llm(prompt)

    return {
        "messages": [AIMessage(content=answer)]
    }


def get_latest_user_message(state: AgentState) -> str:
    messages = state["messages"]

    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message.content

    raise ValueError("No user message found in state.")


def format_conversation_history(state: AgentState) -> str:
    lines = []

    for message in state["messages"]:
        if isinstance(message, HumanMessage):
            lines.append(f"User: {message.content}")
        elif isinstance(message, AIMessage):
            lines.append(f"Assistant: {message.content}")

    return "\n".join(lines)


def search_node(state: AgentState) -> AgentState:
    question = get_latest_user_message(state)

    try:
        record_tool_invocation("search_web")
        logger.info("tool_invoke", extra={"event": "tool_invoke", "tool": "search_web"})

        results = search_web(question)

        tools = list(state.get("tools_invoked") or [])
        tools.append("search_web")
        return {
            "search_results": results,
            "tools_invoked": tools,
        }

    except Exception as exc:
        record_error(component="tool", error_type=type(exc).__name__)
        raise


def answer_with_search_node(state: AgentState) -> AgentState:
    logger.info("answer_with_search_node", extra={"event": "answer_with_search_node"})

    question = get_latest_user_message(state)
    history = format_conversation_history(state)

    prompt = f"""
You are a helpful research assistant.

Use the conversation history and the search results below to answer the user's latest question.
Prefer the conversation history if it already contains the answer.
Use search results only when needed.

Conversation History:
{history}

Latest User Question:
{question}

Search Results:
{state.get('search_results', '')}
""".strip()

    answer = ask_llm(
        prompt,
        system_prompt="You are a helpful research assistant. Use the provided information carefully and answer clearly."
    )

    return {
        "messages": [AIMessage(content=answer)]
    }