"""ReAct-style research agent over arxiv."""

import logging
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.output_parsers import PydanticOutputParser
from langgraph.graph import END, START, StateGraph

from .llm import get_llm
from .schemas import (
    AgentState,
    FinalAnswer,
    Paper,
    PaperReference,
    _AnswerComposition
)
from .tools import arxiv_search

logger = logging.getLogger(__name__)


_MAX_AGENT_STEPS = 6

_AGENT_SYSTEM_PROMPT = (
    "You are an ML research assistant. A user will ask you a question "
    "in natural language. To answer it properly, you must ground your "
    "answer in arxiv papers.\n\n"
    "Your workflow:\n"
    "1. Reformulate the user's question into a concise technical search "
    "query suitable for arxiv (keywords and method names, not prose).\n"
    "2. Call the `arxiv_search` tool with that query.\n"
    "3. Read the abstracts of the returned papers.\n"
    "4. ONLY if the returned abstracts are clearly unrelated to the "
    "question, issue ONE refined search with different keywords.\n"
    "5. As soon as you have at least 2-3 abstracts that together cover "
    "the question, STOP calling tools and write a direct answer in "
    "plain text. Reference papers by title.\n\n"
    "Rules:\n"
    "- Do NOT keep searching for a specific seminal paper. Work with "
    "whatever arxiv returned. Relevance ranking is imperfect; that's ok.\n"
    "- Do NOT issue more than 2 searches in total. Further searches are "
    "almost never useful and waste tokens.\n"
    "- Do not invent facts that are not in the abstracts.\n"
    "- If a tool returns an error, do NOT retry it more than once — "
    "proceed with whatever data you already have.\n"
    "- Write your final answer in the same language as the user's question."
)

_COMPOSE_SYSTEM_PROMPT = (
    "You are assembling a final answer to a user's research question "
    "based on a dialog that already happened and a set of papers that "
    "were retrieved from arxiv.\n\n"
    "Your job:\n"
    "1. Produce a clear, direct answer to the user's question, grounded "
    "in the papers' abstracts.\n"
    "2. For each paper that is actually relevant to the answer, extract "
    "2-4 concrete key ideas from its abstract — not a re-summary of the "
    "abstract, but the specific ideas that relate to the user's question.\n\n"
    "Rules:\n"
    "- Only include papers that are actually relevant. Skip irrelevant "
    "ones retrieved along the way.\n"
    "- Do not invent facts beyond the abstracts.\n"
    "- Use the same language as the user's question."
)


_TOOLS = [arxiv_search]
_TOOLS_BY_NAME = {t.name: t for t in _TOOLS}


def agent_node(state: AgentState) -> dict:
    """One LLM step: decide whether to call a tool or produce a final answer."""
    llm = get_llm().bind_tools(_TOOLS)

    if not state.messages:
        # First step: add system prompt and user question
        messages = [
            SystemMessage(content=_AGENT_SYSTEM_PROMPT),
            HumanMessage(content=state.question),
        ]
        response = llm.invoke(messages)
        logger.info("agent_node: step=0 tool_calls=%s", bool(response.tool_calls))
        return {
            "messages": messages + [response],
            "step_count": 1,
        }
    response = llm.invoke(state.messages)
    logger.info(
        "agent_node: step=%d tool_calls=%s",
        state.step_count,
        bool(response.tool_calls),
    )
    return {"messages": [response], "step_count": state.step_count + 1}


def tool_node(state: AgentState) -> dict:
    """Execute the tool calls from the last AIMessage.

    Tool errors are caught and surfaced back to the LLM as a ToolMessage
    with an ``Error: ...`` prefix, rather than crashing the graph. This
    lets the agent decide whether to retry, try a different query, or
    give up and answer with the data it already has.
    """
    last = state.messages[-1]
    assert isinstance(last, AIMessage) and last.tool_calls, "tool_node called w/o tool_calls"

    new_tool_messages: list[ToolMessage] = []
    new_papers: list[Paper] = []
    seen_ids = {p.arxiv_id for p in state.papers_found}

    for call in last.tool_calls:
        tool = _TOOLS_BY_NAME[call["name"]]
        try:
            result = tool.invoke(call["args"])
        except Exception as exc:
            logger.warning("tool_node: %s failed: %s", call["name"], exc)
            new_tool_messages.append(
                ToolMessage(
                    content=f"Error while calling {call['name']}: {exc}",
                    tool_call_id=call["id"],
                    name=call["name"],
                    status="error",
                )
            )
            continue

        if call["name"] == "arxiv_search":
            for paper in result:
                if paper.arxiv_id not in seen_ids:
                    new_papers.append(paper)
                    seen_ids.add(paper.arxiv_id)
            content = _format_search_result_for_llm(result)
        else:
            content = str(result)

        new_tool_messages.append(
            ToolMessage(content=content, tool_call_id=call["id"], name=call["name"])
        )

    return {
        "messages": new_tool_messages,
        "papers_found": new_papers,
    }


def _format_search_result_for_llm(papers: list[Paper]) -> str:
    """Render a list of papers for the LLM inside a ToolMessage."""
    if not papers:
        return "No papers found for that query."
    lines = [f"Found {len(papers)} papers:"]
    for idx, p in enumerate(papers, 1):
        authors = ", ".join(p.authors[:3])
        if len(p.authors) > 3:
            authors += ", et al."
        lines.append(
            f"\n[{idx}] {p.title}\n"
            f"    arxiv_id: {p.arxiv_id}\n"
            f"    authors: {authors}\n"
            f"    abstract: {p.abstract}"
        )
    return "\n".join(lines)


def should_continue(state: AgentState) -> Literal["tools", "compose"]:
    """After agent_node: go execute tools, or move on to composing the final answer."""
    if state.step_count >= _MAX_AGENT_STEPS:
        logger.warning("should_continue: hit step cap (%d), forcing compose", _MAX_AGENT_STEPS)
        return "compose"

    last = state.messages[-1]

    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "compose"


def compose_node(state: AgentState) -> dict:
    """Build the final FinalAnswer from the dialog and retrieved papers."""
    if not state.papers_found:
        logger.warning("compose_node: no papers retrieved")
        answer = FinalAnswer(
            question=state.question,
            answer="I could not find relevant papers on arxiv for this question.",
            papers=[],
        )
        return {"answer": answer}

    # Last agent text message (without tool_calls) is the answer draft.
    draft = ""
    for m in reversed(state.messages):
        if isinstance(m, AIMessage) and not m.tool_calls and m.content:
            draft = m.content
            break

    parser = PydanticOutputParser(pydantic_object=_AnswerComposition)
    compose_prompt = (
        f"User's question:\n{state.question}\n\n"
        f"Agent's draft answer:\n{draft or '(the agent did not produce a final text; write one yourself)'}\n\n"
        f"Papers retrieved during the dialog:\n{_format_search_result_for_llm(state.papers_found)}\n"
    )

    llm = get_llm()
    messages = [
        SystemMessage(content=_COMPOSE_SYSTEM_PROMPT + "\n\n" + parser.get_format_instructions()),
        HumanMessage(content=compose_prompt),
    ]
    logger.info("compose_node: composing over %d papers", len(state.papers_found))
    composition: _AnswerComposition = (llm | parser).invoke(messages)

    answer = FinalAnswer(
        question=state.question,
        answer=composition.answer,
        papers=composition.references,
    )
    return {"answer": answer}



def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_node("compose", compose_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "compose": "compose",
        },
    )
    graph.add_edge("tools", "agent")
    graph.add_edge("compose", END)
    return graph.compile()
