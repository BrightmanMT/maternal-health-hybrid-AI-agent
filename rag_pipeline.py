from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from openai import OpenAI
from pypdf import PdfReader

from risk_model import RiskModelBundle, build_dataset_context, train_risk_model


PDF_PATH = Path("data/maternal_health.pdf")
LOW_CONFIDENCE_THRESHOLD = 0.34
WEB_MODEL = os.getenv("OPENAI_WEB_MODEL", "gpt-5-mini")
WEB_ALLOWED_DOMAINS = [
    "www.who.int",
    "www.cdc.gov",
    "www.nih.gov",
    "pubmed.ncbi.nlm.nih.gov",
]

LOCAL_PROMPT_TEMPLATE = """
You are a medical assistant helping with maternal health.

Rules:
- Use ONLY the provided context
- Be clear and simple
- If unsure, say you don't know
- Give actionable advice

Context:
{context}

Question:
{question}

Answer:
"""

DATASET_PROMPT_TEMPLATE = """
You are a maternal health data assistant.

Rules:
- Use ONLY the dataset summary provided
- Be concise and practical
- If the dataset summary does not support a conclusion, say you don't know
- When relevant, mention patterns found in the dataset

Dataset summary:
{context}

Question:
{question}

Answer:
"""

LOCAL_PROMPT = PromptTemplate(
    template=LOCAL_PROMPT_TEMPLATE,
    input_variables=["context", "question"],
)

DATASET_PROMPT = PromptTemplate(
    template=DATASET_PROMPT_TEMPLATE,
    input_variables=["context", "question"],
)

GREETING_INPUTS = {
    "hi",
    "hello",
    "hey",
    "good morning",
    "good afternoon",
    "good evening",
}

GREETING_RESPONSE = (
    "Hello! Welcome to the Hybrid Maternal Health Agent. "
    "I can answer maternal-health questions, estimate risk from vitals, and surface WHO guidance."
)

DATASET_HINTS = {
    "dataset",
    "csv",
    "risklevel",
    "risk level",
    "systolicbp",
    "diastolicbp",
    "bodytemp",
    "heartrate",
    "age",
    "distribution",
    "average",
    "mean",
    "median",
    "sample",
    "rows",
    "records",
    "correlation",
}

WEB_FALLBACK_HINTS = {
    "latest",
    "recent",
    "today",
    "current",
    "news",
    "web",
    "internet",
    "online",
}


def _sanitize_proxy_env() -> None:
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        if os.getenv(name) == "http://127.0.0.1:9":
            os.environ.pop(name, None)


@dataclass
class SourceSnippet:
    source_type: str
    title: str
    location: str
    content: str
    url: str | None = None


@dataclass
class RAGResponse:
    answer: str
    sources: list[SourceSnippet]
    answer_mode: str


def _load_pdf_text(pdf_path: Path) -> list[str]:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    reader = PdfReader(str(pdf_path))
    pages: list[str] = []

    for page in reader.pages:
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(text)

    if not pages:
        raise ValueError(f"No readable text found in PDF: {pdf_path}")

    return pages


def _clean_source_text(text: str, max_chars: int = 340) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3].rstrip() + "..."


def _split_pages_with_metadata(
    pages: list[str], chunk_size: int = 900, chunk_overlap: int = 120
) -> list[dict[str, Any]]:
    chunk_records: list[dict[str, Any]] = []

    for page_number, page_text in enumerate(pages, start=1):
        start = 0
        while start < len(page_text):
            end = start + chunk_size
            chunk = page_text[start:end].strip()
            if chunk:
                chunk_records.append({"page": page_number, "content": chunk})

            if end >= len(page_text):
                break

            start = max(end - chunk_overlap, start + 1)

    if not chunk_records:
        raise ValueError("No text chunks were created from the PDF.")

    return chunk_records


def _has_dataset_intent(query: str) -> bool:
    lowered = query.lower()
    return any(hint in lowered for hint in DATASET_HINTS)


def _should_use_web_fallback(query: str, best_score: float, answer: str) -> bool:
    lowered = query.lower()
    if any(hint in lowered for hint in WEB_FALLBACK_HINTS):
        return True
    if best_score < LOW_CONFIDENCE_THRESHOLD:
        return True
    return "don't know" in answer.lower() or "do not know" in answer.lower()


@dataclass
class HybridMaternalAgent:
    embeddings: OpenAIEmbeddings
    llm: ChatOpenAI
    web_client: OpenAI
    index: faiss.IndexFlatIP
    chunks: list[dict[str, Any]]
    guidelines_df: pd.DataFrame
    local_prompt: PromptTemplate
    dataset_prompt: PromptTemplate
    risk_bundle: RiskModelBundle | None = None
    top_k: int = 4

    def run(self, query: str) -> str:
        return self.ask(query).answer

    def ask(self, query: str) -> RAGResponse:
        normalized_query = query.strip().lower()
        if not normalized_query:
            return RAGResponse(answer="", sources=[], answer_mode="empty")

        if normalized_query in GREETING_INPUTS:
            return RAGResponse(
                answer=GREETING_RESPONSE,
                sources=[],
                answer_mode="greeting",
            )

        if _has_dataset_intent(query):
            return self.answer_from_dataset(query)

        local_response, best_score = self.answer_from_pdf(query)
        if _should_use_web_fallback(query, best_score, local_response.answer):
            try:
                return self.answer_from_web(query)
            except Exception:
                return local_response
        return local_response

    def answer_from_dataset(self, query: str) -> RAGResponse:
        context = build_dataset_context(query, bundle=self.risk_bundle)
        prompt_text = self.dataset_prompt.format(context=context, question=query)
        response = self.llm.invoke(prompt_text)
        answer = response.content if isinstance(response.content, str) else str(response.content)
        source = SourceSnippet(
            source_type="dataset",
            title="Maternal Health Risk Data Set",
            location="CSV summary",
            content=_clean_source_text(context),
        )
        return RAGResponse(answer=answer, sources=[source], answer_mode="dataset")

    def answer_from_pdf(self, query: str) -> tuple[RAGResponse, float]:
        query_vector = np.array([self.embeddings.embed_query(query)], dtype="float32")
        faiss.normalize_L2(query_vector)

        k = min(self.top_k, len(self.chunks))
        scores, indices = self.index.search(query_vector, k)
        retrieved_chunks = [
            self.chunks[idx]
            for idx in indices[0]
            if 0 <= idx < len(self.chunks)
        ]

        context = "\n\n".join(chunk["content"] for chunk in retrieved_chunks)
        prompt_text = self.local_prompt.format(context=context, question=query)
        response = self.llm.invoke(prompt_text)
        answer = response.content if isinstance(response.content, str) else str(response.content)

        sources = [
            SourceSnippet(
                source_type="pdf",
                title="WHO Maternal Health PDF",
                location=f"Page {chunk['page']}",
                content=_clean_source_text(chunk["content"]),
            )
            for chunk in retrieved_chunks
        ]

        best_score = float(scores[0][0]) if len(scores[0]) else 0.0
        return RAGResponse(answer=answer, sources=sources, answer_mode="pdf"), best_score

    def answer_from_web(self, query: str) -> RAGResponse:
        response = self.web_client.responses.create(
            model=WEB_MODEL,
            tools=[
                {
                    "type": "web_search",
                    "filters": {"allowed_domains": WEB_ALLOWED_DOMAINS},
                    "search_context_size": "low",
                }
            ],
            include=["web_search_call.action.sources"],
            input=(
                "Answer the maternal health question using reliable web results. "
                "Prefer WHO first, then CDC, NIH, or PubMed. "
                "Be clear, simple, and actionable. "
                f"Question: {query}"
            ),
        )

        sources: list[SourceSnippet] = []
        for item in response.output:
            if item.type != "message":
                continue
            for content in item.content:
                if content.type != "output_text":
                    continue
                for annotation in getattr(content, "annotations", []):
                    if getattr(annotation, "type", "") != "url_citation":
                        continue
                    sources.append(
                        SourceSnippet(
                            source_type="web",
                            title=getattr(annotation, "title", "Web source").strip() or "Web source",
                            location="Web search",
                            content="Information retrieved from a live web source.",
                            url=getattr(annotation, "url", None),
                        )
                    )

        deduped_sources: list[SourceSnippet] = []
        seen_urls: set[str] = set()
        for source in sources:
            if source.url and source.url in seen_urls:
                continue
            if source.url:
                seen_urls.add(source.url)
            deduped_sources.append(source)

        return RAGResponse(
            answer=response.output_text,
            sources=deduped_sources,
            answer_mode="web",
        )

    def get_guidelines_dataframe(self) -> pd.DataFrame:
        return self.guidelines_df.copy()

    def get_dataset_summary(self) -> pd.DataFrame:
        bundle = self.risk_bundle or train_risk_model()
        return bundle.dataset.copy()


def _build_agent(pdf_path: Path, risk_bundle: RiskModelBundle | None = None) -> HybridMaternalAgent:
    _sanitize_proxy_env()

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    pages = _load_pdf_text(pdf_path)
    chunks = _split_pages_with_metadata(pages)
    chunk_texts = [chunk["content"] for chunk in chunks]

    embeddings = OpenAIEmbeddings()
    vectors = np.array(embeddings.embed_documents(chunk_texts), dtype="float32")
    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    guidelines_df = pd.DataFrame(
        {
            "Page": [chunk["page"] for chunk in chunks],
            "Guideline Excerpt": [_clean_source_text(chunk["content"], max_chars=500) for chunk in chunks],
        }
    )

    return HybridMaternalAgent(
        embeddings=embeddings,
        llm=ChatOpenAI(temperature=0, model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")),
        web_client=OpenAI(),
        index=index,
        chunks=chunks,
        guidelines_df=guidelines_df,
        local_prompt=LOCAL_PROMPT,
        dataset_prompt=DATASET_PROMPT,
        risk_bundle=risk_bundle,
    )


@lru_cache(maxsize=1)
def build_rag() -> HybridMaternalAgent:
    return _build_agent(PDF_PATH, train_risk_model())


def build_custom_rag(pdf_path: str | Path, risk_bundle: RiskModelBundle | None = None) -> HybridMaternalAgent:
    return _build_agent(Path(pdf_path), risk_bundle or train_risk_model())
