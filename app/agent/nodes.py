from app.agent.state import AgentState
from app.llm import ask_llm
from app.agent.tools import search_web
from langchain_core.messages import AIMessage, HumanMessage

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

    print(f"[Router] Question: {question}")
    print(f"[Router] Decision: {route}")

    return {
        "route": route
    }


def general_node(state: AgentState) -> AgentState:
    print("[General] Answering directly.")

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

    print(f"[Search] Running search for: {question}")
    results = search_web(question)

    return {
        "search_results": results
    }


def answer_with_search_node(state: AgentState) -> AgentState:
    print("[SearchAnswer] Building final answer from search results.")

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