from typing import Annotated

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from langchain_core.messages import AnyMessage


class Paper(BaseModel):
    arxiv_id: str
    title: str
    url: str
    pdf_url: str
    authors: list[str]
    abstract: str
    published_at: str


class LiteratureReview(BaseModel):
    topic: str
    papers: list[Paper]
    synthesis: str = Field(description="2-3 paragraph synthesis across papers")
    key_concepts: list[str] = Field(description="list of key concepts and methods from the papers")
    open_questions: list[str] = Field(default_factory=list)


class AgentState(BaseModel):
    topic: str
    papers_found: list[Paper] = Field(default_factory=list)
    review: LiteratureReview | None = None
    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)
    step_count: int = 0