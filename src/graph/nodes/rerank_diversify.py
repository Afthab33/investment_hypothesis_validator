"""Rerank and diversify retrieved chunks for balanced evidence."""

from typing import Dict, Any, List
from collections import defaultdict
from src.graph.nodes.base import BaseNode
from src.retrieval.state import IHVState, RetrievedChunk


class RerankDiversify(BaseNode):
    """Rerank chunks and ensure source diversity."""

    def __init__(
        self,
        min_score_threshold: float = 0.01,  # Lowered from 0.3 to work with actual score ranges
        max_per_document: int = 3,
        max_total_chunks: int = 15,
        source_quotas: Dict[str, int] = None
    ):
        """
        Initialize reranking node.

        Args:
            min_score_threshold: Minimum score to keep a chunk
            max_per_document: Maximum chunks from same document
            max_total_chunks: Maximum total chunks to keep
            source_quotas: Minimum chunks per source type
        """
        super().__init__("rerank_diversify", temperature=0.0)
        self.min_score_threshold = min_score_threshold
        self.max_per_document = max_per_document
        self.max_total_chunks = max_total_chunks
        self.source_quotas = source_quotas or {
            "filing": 3,
            "call": 2,
            "chat": 1,
        }

    def process(self, state: IHVState) -> Dict[str, Any]:
        """
        Rerank and diversify chunks.

        Args:
            state: Current state with retrieved_chunks

        Returns:
            State updates with reranked chunks and quality score
        """
        chunks = getattr(state, 'retrieved_chunks', [])
        if not chunks:
            return {
                "retrieved_chunks": [],
                "retrieval_quality": 0.0,
                "needs_rewrite": True
            }

        # Apply score threshold
        filtered_chunks = [
            c for c in chunks
            if c.final_score >= self.min_score_threshold
        ]

        # Ensure source diversity
        diverse_chunks = self._ensure_source_diversity(filtered_chunks)

        # Apply document diversity
        final_chunks = self._apply_document_diversity(diverse_chunks)

        # Calculate retrieval quality
        quality_score = self._calculate_quality_score(final_chunks)

        # Determine if query rewrite is needed
        needs_rewrite = (
            quality_score < 0.4 or
            len(final_chunks) < 5 or
            not self._has_minimum_sources(final_chunks)
        )

        return {
            "retrieved_chunks": final_chunks[:self.max_total_chunks],
            "retrieval_quality": quality_score,
            "needs_rewrite": needs_rewrite
        }

    def _ensure_source_diversity(self, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """
        Ensure minimum representation from each source type.

        Args:
            chunks: Original chunks sorted by score

        Returns:
            Chunks with guaranteed source diversity
        """
        # Group by source type
        by_source = defaultdict(list)
        for chunk in chunks:
            by_source[chunk.source_type].append(chunk)

        # Sort each group by score
        for source in by_source:
            by_source[source].sort(key=lambda x: x.final_score, reverse=True)

        # First pass: fulfill quotas
        selected = []
        for source_type, quota in self.source_quotas.items():
            source_chunks = by_source.get(source_type, [])
            selected.extend(source_chunks[:quota])

        # Second pass: add remaining high-score chunks
        remaining = []
        for source_type, chunks_list in by_source.items():
            quota = self.source_quotas.get(source_type, 0)
            if len(chunks_list) > quota:
                remaining.extend(chunks_list[quota:])

        # Sort remaining by score and add
        remaining.sort(key=lambda x: x.final_score, reverse=True)
        selected.extend(remaining)

        # Remove duplicates while preserving order
        seen_ids = set()
        unique_chunks = []
        for chunk in selected:
            if chunk.chunk_id not in seen_ids:
                seen_ids.add(chunk.chunk_id)
                unique_chunks.append(chunk)

        return unique_chunks

    def _apply_document_diversity(self, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """
        Limit chunks per document to ensure diversity.

        Args:
            chunks: Chunks to filter

        Returns:
            Chunks with document diversity applied
        """
        doc_counts = defaultdict(int)
        filtered = []

        # Sort by score to keep best chunks
        sorted_chunks = sorted(chunks, key=lambda x: x.final_score, reverse=True)

        for chunk in sorted_chunks:
            doc_id = chunk.parent_doc_id
            if doc_counts[doc_id] < self.max_per_document:
                filtered.append(chunk)
                doc_counts[doc_id] += 1

        return filtered

    def _calculate_quality_score(self, chunks: List[RetrievedChunk]) -> float:
        """
        Calculate overall retrieval quality score.

        Args:
            chunks: Final selected chunks

        Returns:
            Quality score between 0 and 1
        """
        if not chunks:
            return 0.0

        # Factor 1: Average chunk score
        avg_score = sum(c.final_score for c in chunks) / len(chunks)

        # Factor 2: Source diversity (0-1)
        unique_sources = len(set(c.source_type for c in chunks))
        source_diversity = unique_sources / 3.0  # Assuming 3 source types

        # Factor 3: Document diversity
        unique_docs = len(set(c.parent_doc_id for c in chunks))
        doc_diversity = min(unique_docs / 5.0, 1.0)  # Cap at 5 different docs

        # Factor 4: Presence of high-confidence chunks
        high_conf_ratio = sum(
            1 for c in chunks if c.final_score > 0.7
        ) / len(chunks)

        # Factor 5: Temporal diversity (different quarters)
        unique_periods = len(set(c.fiscal_period for c in chunks if c.fiscal_period))
        temporal_diversity = min(unique_periods / 3.0, 1.0)  # Cap at 3 quarters

        # Weighted combination
        quality = (
            0.3 * avg_score +
            0.2 * source_diversity +
            0.2 * doc_diversity +
            0.2 * high_conf_ratio +
            0.1 * temporal_diversity
        )

        return min(quality, 1.0)

    def _has_minimum_sources(self, chunks: List[RetrievedChunk]) -> bool:
        """
        Check if we have minimum required sources.

        Args:
            chunks: Retrieved chunks

        Returns:
            True if minimum source requirements are met
        """
        source_counts = defaultdict(int)
        for chunk in chunks:
            source_counts[chunk.source_type] += 1

        # Require at least filing + one other source
        has_filing = source_counts.get("filing", 0) > 0
        has_other = (
            source_counts.get("call", 0) > 0 or
            source_counts.get("chat", 0) > 0
        )

        return has_filing and has_other