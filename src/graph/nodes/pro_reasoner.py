"""PRO reasoning node for finding supporting evidence."""

import json
from typing import Dict, Any, List
from langchain_core.messages import HumanMessage, AIMessage
from src.graph.nodes.base import BaseNode
from src.graph.nodes.utils import get_state_attr
from src.retrieval.state import IHVState, Evidence, RetrievedChunk
from src.prompts.pro_reasoning_prompt import PRO_REASONING_PROMPT


class ProReasoner(BaseNode):
    """Analyze evidence supporting the hypothesis."""

    def __init__(self):
        super().__init__("pro_reasoner", temperature=0.1)

    def process(self, state: IHVState) -> Dict[str, Any]:
        """
        Find and analyze supporting evidence.

        Args:
            state: Current state with retrieved chunks and normalized query

        Returns:
            State updates with pro_evidence
        """
        retrieved_chunks = get_state_attr(state, 'retrieved_chunks', [])
        if not retrieved_chunks:
            return {"pro_evidence": None}

        # Get the hypothesis from the original query
        normalized_query = get_state_attr(state, 'normalized_query')
        hypothesis = normalized_query.original_query if normalized_query else ""
        if not hypothesis:
            # Fallback to last human message
            messages = get_state_attr(state, 'messages', [])
            for message in reversed(messages):
                if isinstance(message, HumanMessage):
                    hypothesis = message.content
                    break

        # Prepare evidence chunks for the prompt
        evidence_text = self._format_evidence_chunks(retrieved_chunks)

        # Create prompt
        prompt = PRO_REASONING_PROMPT.format(
            hypothesis=hypothesis,
            evidence_chunks=evidence_text
        )

        # Get LLM response with retry logic
        from src.aws.bedrock_client import invoke_llm_with_retry
        response = invoke_llm_with_retry(
            messages=[HumanMessage(content=prompt)],
            temperature=self.temperature,
            max_tokens=1500
        )

        # Parse response
        try:
            analysis = self._parse_llm_response(response.content)
        except Exception as e:
            # Fallback parsing if JSON fails
            analysis = self._fallback_parse(response.content)

        # Calculate confidence score
        confidence = self._calculate_confidence(
            analysis.get("supporting_claims", []),
            retrieved_chunks
        )

        # Extract claims and citations
        claims = []
        citations = []
        for claim_obj in analysis.get("supporting_claims", []):
            claim_text = claim_obj.get("claim", "") if isinstance(claim_obj, dict) else str(claim_obj)
            claims.append(claim_text)
            citations.extend(self.extract_citations(claim_text))

        # Create Evidence object
        pro_evidence = Evidence(
            stance="pro",
            claims=claims,
            citations=citations,
            chunks=[c for c in retrieved_chunks if self._chunk_supports(c, claims)],
            confidence=confidence
        )

        return {"pro_evidence": pro_evidence}

    def _format_evidence_chunks(self, chunks: List[RetrievedChunk]) -> str:
        """Format chunks for the prompt."""
        evidence_lines = []
        for i, chunk in enumerate(chunks, 1):
            citation = chunk.get_citation()
            source_info = f"Source: {chunk.source_type.upper()}"
            if chunk.speaker_role:
                source_info += f" - {chunk.speaker} ({chunk.speaker_role})"
            if chunk.fiscal_period:
                source_info += f" - {chunk.fiscal_period}"

            evidence_lines.append(
                f"[{i}] {citation}\n"
                f"{source_info}\n"
                f"Text: {chunk.text[:500]}...\n"
                f"---"
            )

        return "\n".join(evidence_lines)

    def _parse_llm_response(self, response: str) -> dict:
        """Parse LLM JSON response."""
        # Extract JSON from response
        import re
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
        return json.loads(response)

    def _fallback_parse(self, response: str) -> dict:
        """Fallback parsing if JSON fails."""
        # Extract claims manually
        claims = []
        lines = response.split('\n')
        for line in lines:
            if '[' in line and ']' in line:  # Has citation
                claims.append({"claim": line.strip(), "strength": "moderate"})

        return {
            "supporting_claims": claims,
            "key_evidence": "Evidence extracted from response",
            "confidence_rationale": "Fallback parsing used"
        }

    def _calculate_confidence(
        self,
        claims: List[Any],
        chunks: List[RetrievedChunk]
    ) -> float:
        """
        Calculate confidence score for supporting evidence.

        Args:
            claims: List of supporting claims
            chunks: Retrieved chunks used

        Returns:
            Confidence score between 0 and 1
        """
        if not claims:
            return 0.0

        confidence_factors = []

        # Factor 1: Number of strong claims
        strong_claims = sum(
            1 for c in claims
            if isinstance(c, dict) and c.get("strength") == "strong"
        )
        claim_strength = min(strong_claims / max(len(claims), 1), 1.0)
        confidence_factors.append(claim_strength * 0.3)

        # Factor 2: Quantitative vs qualitative evidence
        quantitative_claims = sum(
            1 for c in claims
            if isinstance(c, dict) and c.get("data_type") == "quantitative"
        )
        quant_ratio = quantitative_claims / max(len(claims), 1)
        confidence_factors.append(quant_ratio * 0.25)

        # Factor 3: Source authority
        authority_scores = []
        for chunk in chunks[:10]:  # Top 10 chunks
            weight = self.calculate_source_weight(
                chunk.source_type,
                chunk.speaker_role
            )
            authority_scores.append(weight)
        avg_authority = sum(authority_scores) / max(len(authority_scores), 1)
        confidence_factors.append(avg_authority * 0.25)

        # Factor 4: Source diversity
        unique_sources = len(set(c.source_type for c in chunks[:10]))
        diversity_score = unique_sources / 3.0  # 3 source types
        confidence_factors.append(diversity_score * 0.1)

        # Factor 5: Recency of evidence
        recency_scores = [c.recency_score for c in chunks[:5]]
        avg_recency = sum(recency_scores) / max(len(recency_scores), 1)
        confidence_factors.append(avg_recency * 0.1)

        return min(sum(confidence_factors), 1.0)

    def _chunk_supports(self, chunk: RetrievedChunk, claims: List[str]) -> bool:
        """Check if a chunk supports any of the claims."""
        chunk_citation = chunk.get_citation()
        for claim in claims:
            if chunk_citation in claim or chunk.chunk_id in claim:
                return True
        return False