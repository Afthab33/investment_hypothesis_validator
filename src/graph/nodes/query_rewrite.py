"""Query rewrite node for improving retrieval."""

from typing import Dict, Any
from langchain_core.messages import HumanMessage
from src.graph.nodes.base import BaseNode
from src.retrieval.state import IHVState


class QueryRewriter(BaseNode):
    """Rewrite query when initial retrieval fails."""

    def __init__(self):
        super().__init__("query_rewriter", temperature=0.3)

    def process(self, state: IHVState) -> Dict[str, Any]:
        """
        Rewrite query to improve retrieval.

        Args:
            state: Current state with normalized query

        Returns:
            State updates with rewritten query
        """
        if not state.normalized_query:
            return {}

        original = state.normalized_query.original_query
        normalized = state.normalized_query.normalized_query

        # Build rewrite prompt
        prompt = f"""The following investment query returned poor results.
Please rewrite it with synonyms, related terms, and expanded concepts.

Original: {original}
Normalized: {normalized}

Previous keywords: {', '.join(state.normalized_query.keywords)}

Provide 3 alternative phrasings that might yield better search results:
1. More specific version with financial metrics
2. Broader version capturing the general concept
3. Version using different terminology/synonyms

Format each as a complete question."""

        # Get LLM suggestions
        from src.aws.bedrock_client import invoke_llm_with_retry
        response = invoke_llm_with_retry(messages=[HumanMessage(content=prompt)], temperature=self.temperature, max_tokens=500)

        # Parse rewrites
        rewrites = response.content.strip().split('\n')

        # Take the best rewrite (usually the first one)
        best_rewrite = ""
        for line in rewrites:
            if line.strip() and (line[0].isdigit() or line.startswith('-')):
                # Extract the actual question
                parts = line.split('.', 1)
                if len(parts) > 1:
                    best_rewrite = parts[1].strip()
                    break

        if not best_rewrite:
            best_rewrite = f"{original} including recent trends and management commentary"

        # Update normalized query
        updated_query = state.normalized_query.model_copy()
        updated_query.normalized_query = best_rewrite.lower()

        # Expand keywords
        additional_keywords = [
            "trend", "change", "improvement", "deterioration",
            "quarter over quarter", "year over year",
            "management", "guidance", "outlook"
        ]
        updated_query.keywords.extend(additional_keywords)
        updated_query.keywords = list(set(updated_query.keywords))

        return {
            "normalized_query": updated_query,
            "needs_rewrite": False,  # Prevent infinite loop
            "_rewrite_attempted": True  # Mark rewrite as attempted
        }