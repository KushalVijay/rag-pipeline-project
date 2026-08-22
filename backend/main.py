"""FastAPI entry point for the Web RAG Evaluator."""

import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from backend.rag import (
    KnowledgeBaseError,
    MissingContextDevAPIKeyError,
    MissingOpenAIAPIKeyError,
    RAGInputError,
    WebsiteIngestionError,
    answer_question,
    create_knowledge_base,
    evaluate_case,
    fetch_website_markdown,
)

load_dotenv()

app = FastAPI(
    title="Web RAG Evaluator API",
    description="Small API for a tutorial-sized website RAG system.",
    version="0.1.0",
)

app.state.current_url = None
app.state.current_markdown = None

frontend_origin = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_origin, "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    """Return a lightweight readiness response."""
    return {"status": "ok"}


class IngestRequest(BaseModel):
    """Payload for ingesting one public webpage."""

    model_config = ConfigDict(extra="forbid")

    url: HttpUrl


class IngestResponse(BaseModel):
    """Small ingestion summary returned to the frontend."""

    url: str
    markdown_preview: str
    character_count: int
    chunk_count: int
    title: str | None = None


class AskRequest(BaseModel):
    """Payload for asking one grounded question."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=2_000)


class CitationResponse(BaseModel):
    """One validated source citation used in the answer."""

    label: str
    source_url: str
    chunk_number: int


class RetrievedChunkResponse(BaseModel):
    """One retrieved passage shown alongside an answer."""

    label: str
    text: str
    source_url: str
    chunk_number: int
    distance: float


class AskResponse(BaseModel):
    """Grounded answer and its retrieved evidence."""

    answer: str
    citations: list[CitationResponse]
    retrieved_chunks: list[RetrievedChunkResponse]


class EvaluationCaseRequest(BaseModel):
    """One question and the answer the RAG system should produce."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=2_000)
    expected_answer: str = Field(min_length=1, max_length=4_000)


class EvaluationCaseResponse(BaseModel):
    """Scores for one completed RAG evaluation case."""

    question: str
    expected_answer: str
    generated_answer: str
    retrieval_relevance: float = Field(ge=0, le=1)
    groundedness: float = Field(ge=0, le=1)
    answer_correctness: float = Field(ge=0, le=1)
    citation_validity: float = Field(ge=0, le=1)
    overall_score: float = Field(ge=0, le=1)
    passed: bool


RAG_ERRORS = (RAGInputError, WebsiteIngestionError, KnowledgeBaseError)


def _http_error(error: Exception) -> HTTPException:
    if isinstance(error, RAGInputError):
        return HTTPException(status_code=422, detail=str(error))
    if isinstance(error, (MissingContextDevAPIKeyError, MissingOpenAIAPIKeyError)):
        return HTTPException(status_code=503, detail=str(error))
    return HTTPException(status_code=502, detail=str(error))


@app.post("/api/ingest", response_model=IngestResponse, tags=["knowledge-base"])
def ingest_website(payload: IngestRequest) -> IngestResponse:
    """Ingest one webpage and keep it as the app's current knowledge source."""
    try:
        website = fetch_website_markdown(str(payload.url))
        markdown = website["markdown"]
        chunks = create_knowledge_base(markdown, website["url"])
    except RAG_ERRORS as exc:
        raise _http_error(exc) from exc

    metadata = website["metadata"]

    app.state.current_url = website["url"]
    app.state.current_markdown = markdown

    preview_length = 500
    markdown_preview = markdown[:preview_length]
    if len(markdown) > preview_length:
        markdown_preview += "…"

    return IngestResponse(
        url=website["url"],
        markdown_preview=markdown_preview,
        character_count=len(markdown),
        chunk_count=len(chunks),
        title=metadata.get("title"),
    )


@app.post("/api/ask", response_model=AskResponse, tags=["question-answering"])
def ask_question(payload: AskRequest) -> AskResponse:
    """Answer one question using only the current website knowledge base."""
    try:
        result = answer_question(payload.question)
    except RAG_ERRORS as exc:
        raise _http_error(exc) from exc

    return AskResponse(
        answer=result["answer"],
        citations=result["citations"],
        retrieved_chunks=result["retrieved_chunks"],
    )


@app.post(
    "/api/evaluate",
    response_model=list[EvaluationCaseResponse],
    tags=["evaluations"],
)
def evaluate_cases(
    cases: list[EvaluationCaseRequest],
) -> list[EvaluationCaseResponse]:
    """Run the normal RAG pipeline and evaluator once for every case."""
    if not cases:
        raise HTTPException(
            status_code=422,
            detail="Provide at least one evaluation case.",
        )

    try:
        results = [
            evaluate_case(case.question, case.expected_answer)
            for case in cases
        ]
    except RAG_ERRORS as exc:
        raise _http_error(exc) from exc

    return [
        EvaluationCaseResponse.model_validate(result)
        for result in results
    ]
