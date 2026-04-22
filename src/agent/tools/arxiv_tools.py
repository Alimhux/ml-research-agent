import logging
import arxiv
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ..schemas import Paper

logger = logging.getLogger(__name__)

_MAX_RESULTS_LIMIT = 20
_ARXIV_PAGE_SIZE = 20
_ARXIV_DELAY_SECONDS = 3.0
_ARXIV_NUM_RETRIES = 3


class ArxivSearchInput(BaseModel):
    """Input schema for the arxiv_search tool."""

    query: str = Field(
        description=(
            "Search query for arxiv. Natural language or keywords, "
            "e.g. 'group relative policy optimization' or 'mixture of experts routing'."
        ),
        min_length=1,
    )
    max_results: int = Field(
        default=5,
        ge=1,
        le=_MAX_RESULTS_LIMIT,
        description="Number of papers to return, between 1 and 20.",
    )


@tool("arxiv_search", args_schema=ArxivSearchInput)
def arxiv_search(query: str, max_results: int = 5) -> list[Paper]:
    """Search arxiv for papers matching the query, sorted by relevance.

    Use this tool to find academic papers on a specific ML/AI topic.
    Abstracts are included in the returned papers, so a follow-up fetch
    is usually not needed to build a literature review.

    Args:
        query: Search query for arxiv. Natural language or keywords,
            e.g. "group relative policy optimization".
        max_results: Number of papers to return. Must be between 1 and 20.
            Defaults to 5.

    Returns:
        A list of ``Paper`` objects sorted by arxiv relevance. May be
        empty if no papers match.

    Raises:
        arxiv.UnexpectedEmptyPageError: If arxiv returns a malformed page
            after all retries are exhausted.
        arxiv.HTTPError: If the arxiv API returns a non-retryable HTTP error.
    """

    client = arxiv.Client(
        page_size=_ARXIV_PAGE_SIZE,
        delay_seconds=_ARXIV_DELAY_SECONDS,
        num_retries=_ARXIV_NUM_RETRIES,
    )

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    logger.info("  arxiv_search: query=%r max_results=%d", query, max_results)
    
    papers = [_to_paper(result) for result in client.results(search)]

    logger.info(f"  arxiv_search: found {len(papers)} papers")
    return papers


def _to_paper(result: arxiv.Result) -> Paper:
    """Convert an arxiv.Result into our Paper schema."""
    return Paper(
        arxiv_id=result.get_short_id().split("v")[0],
        title=result.title.strip(),
        url=result.entry_id,
        pdf_url=result.pdf_url,
        authors=[author.name for author in result.authors],
        abstract=result.summary.strip(),
        published_at=result.published.date(),
    )