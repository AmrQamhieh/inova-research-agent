# Inova Research Agent

## Overview

The Inova Research Agent is an AI-powered backend platform developed as part of the Inova Solutions AI & DevOps Internship Program.

The system implements a production-style AI agent architecture capable of:

- Processing user queries through an API
- Routing queries through an AI agent workflow
- Using external tools (web search)
- Maintaining persistent conversation memory
- Running inside a containerized environment
- Providing automated tests and a simple UI interface

The project demonstrates how to build a real-world AI agent backend using modern LLM orchestration frameworks.

---

# System Architecture

User
 ↓
Gradio UI / API Client
 ↓
FastAPI Backend
 ↓
LangGraph Agent Workflow
 ↓
Router Node
 ↓
General LLM Response OR Search Tool
 ↓
PostgreSQL Conversation Memory

The agent decides whether a query can be answered directly or requires external information.

---

# Features

AI Agent Workflow
- Built using LangGraph
- Intelligent query routing
- Tool usage for external data retrieval

Persistent Memory
- Conversation history stored in PostgreSQL
- Thread-based conversation tracking
- Ability to retrieve full conversation history

API Backend
- FastAPI REST API
- Structured request and response models

Web Interface
- Simple Gradio UI for interacting with the agent

Containerized Infrastructure
- Dockerized application
- PostgreSQL database container
- Docker Compose orchestration

Automated Testing
- Pytest test suite
- API endpoint testing
- Agent routing logic testing

---

# Tech Stack

Backend
- Python
- FastAPI
- Pydantic

AI Layer
- OpenAI API
- LangGraph

Data Layer
- PostgreSQL
- SQLAlchemy

Infrastructure
- Docker
- Docker Compose

Testing
- Pytest
- pytest-mock

Interface
- Gradio

---

# Project Structure

inova-research-agent

app/
  main.py
  database.py
  schemas.py
  agent/
    graph.py
    nodes.py

tests/
  test_main.py
  test_router_node.py

gradio_app.py
docker-compose.yml
Dockerfile
requirements.txt
.env.example
README.md

---

# API Endpoints

GET /
Returns a simple message confirming the API is running.

GET /health
Health check endpoint to verify the backend service.

POST /query
Processes a user query through the AI agent workflow.

Example Request:
{
  "thread_id": "demo1",
  "text": "Explain what FastAPI is."
}

Example Response:
{
  "thread_id": "demo1",
  "response": "FastAPI is a modern Python web framework..."
}

GET /history/{thread_id}
Retrieves the full conversation history for a given thread.

---

# Running the Project Locally

1. Clone the repository

git clone https://github.com/AmrQamhieh/inova-research-agent.git
cd inova-research-agent
git checkout develop

2. Create a virtual environment

python -m venv venv

Activate:

Windows
venv\\Scripts\\activate

Linux / Mac
source venv/bin/activate

3. Install dependencies

pip install -r requirements.txt

4. Configure environment variables

Create a .env file based on .env.example

Example:

OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini

5. Run the API

uvicorn app.main:app --reload

API Docs
http://127.0.0.1:8000/docs

---

# Running with Docker

docker compose up --build

This starts:
- FastAPI backend
- PostgreSQL database

---

# Running Tests

python -m pytest

Test coverage includes:
- API endpoint validation
- Query endpoint behavior
- Router node decision logic
- Conversation history endpoint

---

# Development Workflow

git checkout develop
git pull origin develop
git add .
git commit -m "feat: implement feature"
git push origin develop

---

# Contributors

Khaled Yaish – AI & Backend Development
Amr Qamhieh – DevOps & Infrastructure

---

# License

Developed as part of the Inova Solutions AI & DevOps Internship Program for educational and research purposes.
"""