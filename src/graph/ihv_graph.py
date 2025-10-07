"""Main LangGraph workflow for Investment Hypothesis Validator."""

from typing import Dict, Any
from langgraph.graph import StateGraph, END
from src.retrieval.state import IHVState
from src.graph.nodes import (
    QuestionNormalizer,
    RerankDiversify,
    ProReasoner,
    ConReasoner,
    VerdictSynthesizer,
    QueryRewriter,
    ReportFormatter,
    ToneDeltaAnalyzer,
)
from src.retrieval.hybrid_retriever import StratifiedHybridRetriever


class RetrievalNode:
    """Wrapper node for the retrieval layer."""

    def __init__(self):
        self.name = "retrieval"
        self.retriever = StratifiedHybridRetriever(
            k_per_source=5,
            vector_weight=0.6,
            keyword_weight=0.4,
        )

    def __call__(self, state: IHVState) -> Dict[str, Any]:
        """Execute retrieval."""
        # Handle state as dict
        if isinstance(state, dict):
            normalized_query = state.get("normalized_query")
        else:
            normalized_query = state.normalized_query

        if not normalized_query:
            return {"retrieved_chunks": []}

        # Extract filters from normalized query
        ticker = normalized_query.ticker
        fiscal_period = normalized_query.fiscal_period

        # Perform retrieval with expanded query
        chunks = self.retriever.retrieve(
            query=normalized_query.normalized_query,
            ticker=ticker,
            fiscal_period=fiscal_period,
        )

        return {"retrieved_chunks": chunks}


def should_rewrite(state: IHVState) -> str:
    """Determine if query rewrite is needed."""
    # Handle state as dict
    if isinstance(state, dict):
        needs_rewrite = state.get('needs_rewrite', False)
        retrieved_chunks = state.get('retrieved_chunks')
        rewrite_attempted = state.get('_rewrite_attempted', False)
    else:
        needs_rewrite = getattr(state, 'needs_rewrite', False)
        retrieved_chunks = getattr(state, 'retrieved_chunks', None)
        rewrite_attempted = getattr(state, '_rewrite_attempted', False)

    if needs_rewrite and retrieved_chunks is not None and not rewrite_attempted:
        return "rewrite"
    return "continue"


def create_ihv_graph() -> StateGraph:
    """
    Create the Investment Hypothesis Validator graph.

    Returns:
        Configured LangGraph StateGraph
    """
    # Initialize graph with state schema
    workflow = StateGraph(IHVState)

    # Add nodes
    workflow.add_node("normalize", QuestionNormalizer())
    workflow.add_node("retrieve", RetrievalNode())
    workflow.add_node("rerank", RerankDiversify())
    workflow.add_node("rewrite", QueryRewriter())
    workflow.add_node("pro_reason", ProReasoner())
    workflow.add_node("con_reason", ConReasoner())
    workflow.add_node("synthesize", VerdictSynthesizer())
    workflow.add_node("tone_delta", ToneDeltaAnalyzer())
    workflow.add_node("format", ReportFormatter())

    # Add edges
    workflow.add_edge("normalize", "retrieve")

    # Conditional edge for query rewrite
    workflow.add_conditional_edges(
        "retrieve",
        lambda state: "rerank",  # For now, always go to rerank
        {
            "rerank": "rerank",
            "rewrite": "rewrite"
        }
    )

    # After rerank, check if rewrite is needed
    workflow.add_conditional_edges(
        "rerank",
        should_rewrite,
        {
            "continue": "pro_reason",
            "rewrite": "rewrite"
        }
    )

    # Rewrite goes back to retrieve
    workflow.add_edge("rewrite", "retrieve")

    # Sequential reasoning to avoid Bedrock throttling
    # PRO reasoning first, then CON reasoning
    workflow.add_edge("pro_reason", "con_reason")

    # CON reasoner goes to synthesizer
    workflow.add_edge("con_reason", "synthesize")

    # Synthesizer to tone delta analyzer
    workflow.add_edge("synthesize", "tone_delta")

    # Tone delta to formatter
    workflow.add_edge("tone_delta", "format")

    # Formatter is the end
    workflow.add_edge("format", END)

    # Set entry point
    workflow.set_entry_point("normalize")

    return workflow.compile()