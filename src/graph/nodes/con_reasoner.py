"""CON reasoning node for finding contradicting evidence."""

import json
from typing import Dict, Any, List
from langchain_core.messages import HumanMessage, AIMessage
from src.graph.nodes.base import BaseNode
from src.graph.nodes.utils import get_state_attr
from src.retrieval.state import IHVState, Evidence, RetrievedChunk
from src.prompts.con_reasoning_prompt import CON_REASONING_PROMPT


class ConReasoner(BaseNode):
    """Analyze evidence contradicting the hypothesis."""

    def __init__(self):
        super().__init__("con_reasoner", temperature=0.1)

    def process(self, state: IHVState) -> Dict[str, Any]:
        """
        Find and analyze contradicting evidence.

        Args:
            state: Current state with retrieved chunks and normalized query

        Returns:
            State updates with con_evidence
        """
        retrieved_chunks = get_state_attr(state, "retrieved_chunks", [])
        if not retrieved_chunks:
            return {"con_evidence": None}

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
        prompt = CON_REASONING_PROMPT.format(
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
            analysis.get("refuting_claims", []),
            retrieved_chunks
        )

        # Extract claims and citations
        claims = []
        citations = []
        for claim_obj in analysis.get("refuting_claims", []):
            claim_text = claim_obj.get("claim", "") if isinstance(claim_obj, dict) else str(claim_obj)
            claims.append(claim_text)
            citations.extend(self.extract_citations(claim_text))

        # Create Evidence object
        con_evidence = Evidence(
            stance="con",
            claims=claims,
            citations=citations,
            chunks=[c for c in retrieved_chunks if self._chunk_refutes(c, claims)],
            confidence=confidence
        )

        return {"con_evidence": con_evidence}

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

            # Include sentiment if available
            if chunk.sentiment:
                source_info += f" - Sentiment: {chunk.sentiment}"

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
                claims.append({
                    "claim": line.strip(),
                    "strength": "moderate",
                    "refutation_type": "direct"
                })

        return {
            "refuting_claims": claims,
            "key_concerns": "Concerns extracted from response",
            "confidence_rationale": "Fallback parsing used"
        }

    def _calculate_confidence(
        self,
        claims: List[Any],
        chunks: List[RetrievedChunk]
    ) -> float:
        """
        Calculate confidence score for refuting evidence.

        Args:
            claims: List of refuting claims
            chunks: Retrieved chunks used

        Returns:
            Confidence score between 0 and 1
        """
        if not claims:
            return 0.0

        confidence_factors = []

        # Factor 1: Number and strength of direct refutations
        direct_refutations = sum(
            1 for c in claims
            if isinstance(c, dict) and c.get("refutation_type") == "direct"
        )
        strong_refutations = sum(
            1 for c in claims
            if isinstance(c, dict) and c.get("strength") == "strong"
        )
        refutation_strength = (
            (direct_refutations * 0.6 + strong_refutations * 0.4) /
            max(len(claims), 1)
        )
        confidence_factors.append(refutation_strength * 0.35)

        # Factor 2: Negative sentiment in evidence
        negative_chunks = [
            c for c in chunks[:10]
            if c.sentiment == "bearish" or
            (c.sentiment_score and c.sentiment_score < -0.3)
        ]
        negative_ratio = len(negative_chunks) / min(len(chunks), 10)
        confidence_factors.append(negative_ratio * 0.2)

        # Factor 3: Source authority (higher weight for official filings)
        authority_scores = []
        for chunk in chunks[:10]:
            # Give extra weight to risk factors and warnings
            weight = self.calculate_source_weight(
                chunk.source_type,
                chunk.speaker_role
            )
            if chunk.section and "risk" in chunk.section.lower():
                weight *= 1.2
            authority_scores.append(weight)
        avg_authority = sum(authority_scores) / max(len(authority_scores), 1)
        confidence_factors.append(avg_authority * 0.25)

        # Factor 4: Presence of quantitative negative evidence
        quantitative_claims = sum(
            1 for c in claims[:5]
            if isinstance(c, dict) and
            any(char.isdigit() for char in c.get("claim", ""))
        )
        quant_score = min(quantitative_claims / 3, 1.0)
        confidence_factors.append(quant_score * 0.15)

        # Factor 5: Consistency (multiple sources saying same thing)
        if len(claims) > 1:
            unique_sources = len(set(
                c.source_type for c in chunks
                if self._chunk_refutes(c, [str(claim) for claim in claims])
            ))
            consistency_score = min(unique_sources / 2, 1.0)
            confidence_factors.append(consistency_score * 0.05)

        return min(sum(confidence_factors), 1.0)

    def _chunk_refutes(self, chunk: RetrievedChunk, claims: List[str]) -> bool:
        """Check if a chunk supports any of the refuting claims."""
        chunk_citation = chunk.get_citation()
        for claim in claims:
            if chunk_citation in claim or chunk.chunk_id in claim:
                return True
        # Also check for negative sentiment chunks
        return (
            chunk.sentiment == "bearish" or
            (chunk.sentiment_score and chunk.sentiment_score < -0.2)
        )