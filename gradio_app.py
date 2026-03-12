import gradio as gr
import requests

API_BASE_URL = "http://127.0.0.1:8000"


def send_message(message, thread_id, chat_history):
    chat_history = chat_history or []

    if not message or not message.strip():
        return chat_history, ""

    payload = {
        "thread_id": thread_id,
        "text": message
    }

    try:
        response = requests.post(f"{API_BASE_URL}/query", json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()

        bot_reply = data.get("response", "No response returned.")

        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": bot_reply})

        return chat_history, ""

    except Exception as e:
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": f"Error: {str(e)}"})
        return chat_history, ""


def load_history(thread_id):
    try:
        response = requests.get(f"{API_BASE_URL}/history/{thread_id}", timeout=30)
        response.raise_for_status()
        data = response.json()

        messages = data.get("messages", [])
        chat_history = []

        for msg in messages:
            question = msg.get("question", "")
            answer = msg.get("response", "")

            if question:
                chat_history.append({"role": "user", "content": question})
            if answer:
                chat_history.append({"role": "assistant", "content": answer})

        return chat_history

    except Exception as e:
        return [{"role": "assistant", "content": f"Failed to load history: {str(e)}"}]


def clear_chat():
    return []


with gr.Blocks() as demo:
    gr.Markdown("# Inova Research Agent")
    gr.Markdown("Simple frontend for chatting with the LangGraph agent.")

    thread_id = gr.Textbox(label="Thread ID", value="demo1")
    chatbot = gr.Chatbot(label="Conversation", height=400)
    message = gr.Textbox(label="Your Message", placeholder="Ask something...")

    with gr.Row():
        send_btn = gr.Button("Send")
        load_btn = gr.Button("Load History")
        clear_btn = gr.Button("Clear Chat")

    send_btn.click(
        fn=send_message,
        inputs=[message, thread_id, chatbot],
        outputs=[chatbot, message]
    )

    message.submit(
        fn=send_message,
        inputs=[message, thread_id, chatbot],
        outputs=[chatbot, message]
    )

    load_btn.click(
        fn=load_history,
        inputs=[thread_id],
        outputs=[chatbot]
    )

    clear_btn.click(
        fn=clear_chat,
        inputs=[],
        outputs=[chatbot]
    )

demo.launch()