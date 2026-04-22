from datetime import date
from typing import Annotated
from operator import add

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class Paper(BaseModel):
    """A single arxiv paper retrieved during search."""
    arxiv_id: str = Field(description="Arxiv short id without version suffix.")
    title: str
    url: str = Field(description="URL of the arxiv abstract page.")
    pdf_url: str = Field(description="Direct URL to the paper PDF.")
    authors: list[str]
    abstract: str
    published_at: date


class PaperReference(BaseModel):
    """A paper cited in the final answer, with LLM-extracted key ideas."""
    arxiv_id: str
    title: str
    url: str
    abstract: str
    key_ideas: list[str] = Field(
        description="Concrete ideas from this paper that are relevant to the user's question."
    )


class FinalAnswer(BaseModel):
    question: str = Field(description="The user's original question.")
    answer: str = Field(
        description="A direct answer to the question, grounded in the referenced papers."
    )
    papers: list[PaperReference] = Field(
        default_factory=list,
        description="Papers used to support the answer."
    )


# LLM-facing sub-schema for the compose step
class _AnswerComposition(BaseModel):
    """Output of the compose node: answer + per-paper key ideas."""
    answer: str = Field(description="Direct answer to the question.")
    references: list[PaperReference] = Field(
        description="Papers relevant to the answer, with key ideas extracted per paper."
    )


class AgentState(BaseModel):
    """Mutable state passed between graph nodes."""
    question: str = Field(description="The user's original natural-language question.")
    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)
    papers_found: Annotated[list[Paper], add] = Field(default_factory=list)
    answer: FinalAnswer | None = None
    step_count: int = 0