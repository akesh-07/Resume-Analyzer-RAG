"""Core RAG pipeline for resume analysis against job descriptions."""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import BinaryIO

import pdfplumber
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI


@dataclass
class ResumeAnalysis:
    resume_name: str
    evaluation: str


def extract_text_from_pdf(source: str | BinaryIO) -> str:
    """Extract plain text from a PDF file path or file-like object."""
    text = ""
    with pdfplumber.open(source) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()


def extract_text_from_bytes(file_bytes: bytes) -> str:
    return extract_text_from_pdf(io.BytesIO(file_bytes))


def build_vector_store(resume_data: dict[str, str]) -> FAISS:
    """Chunk resumes and build a FAISS vector store."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )

    documents: list[Document] = []
    for file_name, text in resume_data.items():
        for chunk in splitter.split_text(text):
            documents.append(
                Document(page_content=chunk, metadata={"source": file_name})
            )

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.from_documents(documents, embedding_model)


def create_llm(
    model: str = "gemini-2.5-flash",
    temperature: float = 0.2,
    api_key: str | None = None,
) -> ChatGoogleGenerativeAI:
    key = api_key or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise ValueError(
            "Google API key is required. Set GOOGLE_API_KEY in your environment "
            "or provide it in the app sidebar."
        )
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        google_api_key=key,
    )


def analyze_resume(
    resume_name: str,
    job_description: str,
    vector_db: FAISS,
    llm: ChatGoogleGenerativeAI,
    top_k: int = 5,
) -> str:
    """Retrieve relevant resume chunks and evaluate against the job description."""
    docs = vector_db.similarity_search(
        job_description,
        k=top_k,
        filter={"source": resume_name},
    )
    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = f"""
You are an ATS resume evaluator.

JOB DESCRIPTION:
{job_description}

RESUME:
{context}

Evaluate the resume.

Return:
1. Match Percentage
2. Matching Skills
3. Missing Skills
4. Strengths
5. Weaknesses
6. Final Recommendation

Give response in proper format.
"""
    response = llm.invoke(prompt)
    return response.content


def analyze_all_resumes(
    resume_data: dict[str, str],
    job_description: str,
    model: str = "gemini-2.5-flash",
    temperature: float = 0.2,
    api_key: str | None = None,
) -> list[ResumeAnalysis]:
    """Run the full pipeline: embed, retrieve, and evaluate each resume."""
    vector_db = build_vector_store(resume_data)
    llm = create_llm(model=model, temperature=temperature, api_key=api_key)

    results: list[ResumeAnalysis] = []
    for resume_name in resume_data:
        evaluation = analyze_resume(resume_name, job_description, vector_db, llm)
        results.append(ResumeAnalysis(resume_name=resume_name, evaluation=evaluation))
    return results
