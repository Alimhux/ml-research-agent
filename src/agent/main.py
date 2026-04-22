"""CLI entrypoint for the research agent.

Run from the project root with the package installed in editable mode:

    python -m agent.main "Что такое трансформер и как декодер взаимодействует с энкодером?"
    python -m agent.main -v "What is GRPO?"

The result is printed to stdout as JSON-serialized ``FinalAnswer``.
"""

import argparse
import logging
import sys

from .graph import build_graph
from .schemas import AgentState, FinalAnswer

logger = logging.getLogger("agent")


_DEFAULT_QUESTION = (
    "Что такое трансформер и как декодер взаимодействует с энкодером?"
)


def run_research(question: str) -> FinalAnswer:
    """Run the research agent graph end-to-end for a user question.

    Args:
        question: The user's natural-language research question.

    Returns:
        A ``FinalAnswer`` with a direct answer and a list of cited papers.

    Raises:
        RuntimeError: If the graph completes without producing an answer.
    """
    graph = build_graph()
    final = graph.invoke(AgentState(question=question))
    answer = final.get("answer")
    if answer is None:
        raise RuntimeError("graph finished without producing an answer")
    return answer


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ask a natural-language research question, get an answer "
            "grounded in arxiv papers."
        ),
    )
    parser.add_argument(
        "question",
        nargs="?",
        default=_DEFAULT_QUESTION,
        help="Your research question. Defaults to a transformer demo query.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable INFO-level logging on stderr.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI main. Returns a process exit code."""
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )
    answer = run_research(args.question)
    print(answer.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
