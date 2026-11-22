"""LangGraph workflow definition.

    START -> planner --DIRECT_ANSWER--> direct_answer ----------------------> respond -> END
                     --CLARIFICATION--> clarify ----------------------------> respond
                     --RAG_RETRIEVAL--> retrieve -> grade --sufficient-----> generate -> respond
                                           ^           |--insufficient & attempts < max--> rewrite
                                           |___________________________________________|
                                                       |--insufficient & attempts == max --> generate
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from backend.agent.nodes import AgentNodes, Retriever
from backend.agent.state import AgentState


def build_graph(router_llm: Any, answer_llm: Any, retriever: Retriever, *,
                top_k: int = 6, max_attempts: int = 2, history_turns: int = 6, probe_library: bool = True):
    nodes = AgentNodes(router_llm, answer_llm, retriever, top_k=top_k,
                       max_attempts=max_attempts, history_turns=history_turns, probe_library=probe_library)

    g = StateGraph(AgentState)
    g.add_node("planner", nodes.planner)
    g.add_node("retrieve", nodes.retrieve)
    g.add_node("grade", nodes.grade)
    g.add_node("rewrite", nodes.rewrite)
    g.add_node("generate", nodes.generate)
    g.add_node("direct_answer", nodes.direct_answer)
    g.add_node("clarify", nodes.clarify)
    g.add_node("respond", nodes.respond)

    g.add_edge(START, "planner")
    g.add_conditional_edges("planner", nodes.route_after_planner,
                            {"retrieve": "retrieve", "direct_answer": "direct_answer", "clarify": "clarify"})
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", nodes.route_after_grade,
                            {"generate": "generate", "rewrite": "rewrite"})
    g.add_edge("rewrite", "retrieve")
    for n in ("generate", "direct_answer", "clarify"):
        g.add_edge(n, "respond")
    g.add_edge("respond", END)
    return g.compile()
