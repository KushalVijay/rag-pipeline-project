# Web RAG Evaluator

Web RAG Evaluator ingests one public webpage, answers questions from that page, and scores the RAG result. It is intentionally small enough to explain in a short tutorial.

## Tools used

- Python, FastAPI, Context.dev, OpenAI, and local persistent Chroma
- React, Vite, and TypeScript

## Architecture

- `backend/main.py` contains the FastAPI routes, request models, CORS setup, and in-memory current-page state.
- `backend/rag.py` contains Context.dev ingestion, Markdown chunking, OpenAI embeddings and responses, local persistent Chroma retrieval, citation checks, and evaluation scoring.
- `frontend/src/App.tsx` and `frontend/src/index.css` contain the three-tab React interface.
- API keys stay in the backend `.env`; the frontend calls only the FastAPI API.

## Setup

From the project root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Fill in `CONTEXT_DEV_API_KEY` and `OPENAI_API_KEY` in `.env`, then start the backend:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload --port 8000
```

In a second terminal, start the frontend:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. The backend health endpoint is `http://localhost:8000/health`.

## RAG pipeline

Context.dev turns the URL into main-content Markdown. The backend splits that Markdown near headings and paragraph boundaries, creates OpenAI embeddings in batches, and replaces the single `website_knowledge` Chroma collection. A question is embedded, the four closest chunks are retrieved, and the OpenAI Responses API answers only from those labeled chunks. Citations are filtered so only supplied labels can be returned.

## Evaluations

Each case runs the normal answer pipeline. One structured OpenAI evaluator call scores retrieval relevance, groundedness, and answer correctness from 0 to 1. Citation validity is checked deterministically. The four scores are averaged, and a case passes at `0.70` or higher.

## Known limitation

This MVP supports one webpage at a time. Ingesting another URL replaces the existing Chroma collection and current in-memory page state.
