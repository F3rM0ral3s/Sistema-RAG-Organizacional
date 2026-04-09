"""Pydantic models for request/response schemas."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class QueryStatus(str, Enum):
    PROCESSING = "processing"
    PROCESSED = "processed"
    REJECTED = "rejected"


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    user_id: str = Field(..., min_length=1, max_length=128)


class QuerySubmitResponse(BaseModel):
    query_id: str
    status: QueryStatus


class SourceChunk(BaseModel):
    text: str
    doc_id: str = ""
    chunk_id: str = ""
    corpus: str = ""
    decade: str = ""
    issue_date: str = ""
    source_pdf: str = ""
    chunk_index: int = 0
    score: float = 0.0


class QueryResultResponse(BaseModel):
    query_id: str
    status: QueryStatus
    answer: Optional[str] = None
    sources: list[SourceChunk] = []
    expanded_queries: list[str] = []
    rejection_reason: Optional[str] = None
