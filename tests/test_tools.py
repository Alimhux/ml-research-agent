"""Tests for agent.tools.arxiv_tools."""

from datetime import date, datetime, timezone
from types import SimpleNamespace

import pytest

from agent.schemas import Paper
from agent.tools.arxiv_tools import _to_paper, arxiv_search


def _fake_result(**overrides) -> SimpleNamespace:
    """Build a stand-in for an arxiv.Result with sensible defaults."""
    defaults = dict(
        title="  Attention Is All You Need  ",
        summary=" A transformer-based sequence model. ",
        authors=[SimpleNamespace(name=n) for n in ["Vaswani", "Shazeer"]],
        entry_id="http://arxiv.org/abs/1706.03762v5",
        pdf_url="http://arxiv.org/pdf/1706.03762v5",
        published=datetime(2017, 6, 12, tzinfo=timezone.utc),
    )
    defaults.update(overrides)
    ns = SimpleNamespace(**defaults)
    ns.get_short_id = lambda: "1706.03762v5"
    return ns


class TestToPaper:
    """Unit tests for the arxiv.Result -> Paper mapping."""

    def test_maps_all_fields(self):
        paper = _to_paper(_fake_result())
        assert paper.arxiv_id == "1706.03762"
        assert paper.title == "Attention Is All You Need"
        assert paper.abstract == "A transformer-based sequence model."
        assert paper.authors == ["Vaswani", "Shazeer"]
        assert paper.url == "http://arxiv.org/abs/1706.03762v5"
        assert paper.pdf_url == "http://arxiv.org/pdf/1706.03762v5"
        assert paper.published_at == date(2017, 6, 12)

    def test_strips_version_suffix_from_arxiv_id(self):
        result = _fake_result()
        result.get_short_id = lambda: "2402.03300v3"
        assert _to_paper(result).arxiv_id == "2402.03300"

    def test_returns_paper_instance(self):
        assert isinstance(_to_paper(_fake_result()), Paper)


@pytest.mark.integration
class TestArxivSearch:
    """Integration tests for arxiv_search (hit the live arxiv API)."""

    def test_returns_requested_number_of_papers(self):
        papers = arxiv_search.invoke(
            {"query": "attention is all you need", "max_results": 3}
        )
        assert len(papers) == 3
        for p in papers:
            assert isinstance(p, Paper)
            assert p.title
            assert p.abstract
            assert p.arxiv_id
