"""Verdict synthesis node for final decision making."""

from typing import Dict, Any, List, Optional
from src.graph.nodes.base import BaseNode
from src.graph.nodes.utils import get_state_attr
from src.retrieval.state import IHVState, Verdict, Evidence


class VerdictSynthesizer(BaseNode):
    """Synthesize PRO and CON evidence into final verdict."""

    def __init__(
        self,
        support_threshold: float = 0.6,
        refute_threshold: float = 0.6,
        min_evidence_threshold: float = 0.3
    ):
        """
        Initialize verdict synthesizer.

        Args:
            support_threshold: Min PRO confidence for Support verdict
            refute_threshold: Min CON confidence for Refute verdict
            min_evidence_threshold: Min confidence for any verdict
        """
        super().__init__("verdict_synthesizer", temperature=0.0)
        self.support_threshold = support_threshold
        self.refute_threshold = refute_threshold
        self.min_evidence_threshold = min_evidence_threshold

    def process(self, state: IHVState) -> Dict[str, Any]:
        """
        Synthesize final verdict from PRO and CON evidence.

        Args:
            state: Current state with pro_evidence and con_evidence

        Returns:
            State updates with verdict
        """
        pro_evidence = get_state_attr(state, 'pro_evidence')
        con_evidence = get_state_attr(state, 'con_evidence')

        # Handle missing evidence
        if not pro_evidence and not con_evidence:
            verdict = Verdict(
                verdict="Inconclusive",
                confidence=0.0,
                rationale=["No evidence found to evaluate the hypothesis"],
                counterpoints=[],
                pro_evidence=None,
                con_evidence=None
            )
            return {"verdict": verdict}

        # Extract confidence scores
        pro_confidence = pro_evidence.confidence if pro_evidence else 0.0
        con_confidence = con_evidence.confidence if con_evidence else 0.0

        # Determine verdict
        verdict_type, overall_confidence = self._determine_verdict(
            pro_confidence,
            con_confidence
        )

        # Build rationale
        rationale = self._build_rationale(pro_evidence, con_evidence, verdict_type)

        # Extract counterpoints
        counterpoints = self._extract_counterpoints(
            pro_evidence,
            con_evidence,
            verdict_type
        )

        # Create Verdict object
        verdict = Verdict(
            verdict=verdict_type,
            confidence=overall_confidence,
            rationale=rationale,
            counterpoints=counterpoints,
            tone_delta=None,  # Will be set by tone delta node
            pro_evidence=pro_evidence,
            con_evidence=con_evidence
        )

        return {"verdict": verdict}

    def _determine_verdict(
        self,
        pro_confidence: float,
        con_confidence: float
    ) -> tuple[str, float]:
        """
        Determine verdict based on confidence scores.

        Args:
            pro_confidence: PRO evidence confidence
            con_confidence: CON evidence confidence

        Returns:
            Tuple of (verdict_type, overall_confidence)
        """
        # Check for insufficient evidence
        if pro_confidence < self.min_evidence_threshold and \
           con_confidence < self.min_evidence_threshold:
            return "Inconclusive", min(pro_confidence, con_confidence)

        # Strong support case
        if pro_confidence >= self.support_threshold and \
           pro_confidence > con_confidence * 1.5:
            confidence = pro_confidence * (1 - con_confidence * 0.3)
            return "Support", confidence

        # Strong refute case
        if con_confidence >= self.refute_threshold and \
           con_confidence > pro_confidence * 1.5:
            confidence = con_confidence * (1 - pro_confidence * 0.3)
            return "Refute", confidence

        # Mixed evidence cases
        if abs(pro_confidence - con_confidence) < 0.2:
            # Too close to call
            return "Inconclusive", max(pro_confidence, con_confidence) * 0.5

        # Lean towards higher confidence but not strong enough
        if pro_confidence > con_confidence:
            # Weak support
            confidence = pro_confidence * 0.7
            if confidence > 0.5:
                return "Support", confidence
        else:
            # Weak refute
            confidence = con_confidence * 0.7
            if confidence > 0.5:
                return "Refute", confidence

        # Default to inconclusive
        return "Inconclusive", (pro_confidence + con_confidence) / 2

    def _build_rationale(
        self,
        pro_evidence: Optional[Evidence],
        con_evidence: Optional[Evidence],
        verdict: str
    ) -> List[str]:
        """
        Build rationale statements for the verdict.

        Args:
            pro_evidence: Supporting evidence
            con_evidence: Refuting evidence
            verdict: Final verdict type

        Returns:
            List of rationale statements with citations
        """
        rationale = []

        if verdict == "Support":
            # Lead with strongest supporting claims
            if pro_evidence:
                # Take top 3 claims
                for claim in pro_evidence.claims[:3]:
                    rationale.append(claim)

                # Add confidence qualifier
                if pro_evidence.confidence > 0.8:
                    rationale.append("Strong consensus across multiple sources")
                elif pro_evidence.confidence > 0.6:
                    rationale.append("Moderate support from available evidence")

        elif verdict == "Refute":
            # Lead with strongest refuting claims
            if con_evidence:
                # Take top 3 claims
                for claim in con_evidence.claims[:3]:
                    rationale.append(claim)

                # Add confidence qualifier
                if con_evidence.confidence > 0.8:
                    rationale.append("Significant contradicting evidence found")
                elif con_evidence.confidence > 0.6:
                    rationale.append("Multiple concerns identified in the data")

        else:  # Inconclusive
            # Balance both sides
            if pro_evidence and pro_evidence.claims:
                rationale.append(f"Supporting: {pro_evidence.claims[0]}")
            if con_evidence and con_evidence.claims:
                rationale.append(f"However: {con_evidence.claims[0]}")

            # Add reason for inconclusiveness
            if not pro_evidence or pro_evidence.confidence < 0.3:
                rationale.append("Insufficient supporting evidence found")
            elif not con_evidence or con_evidence.confidence < 0.3:
                rationale.append("Limited contradicting evidence to fully refute")
            else:
                rationale.append("Mixed signals prevent definitive conclusion")

        return rationale

    def _extract_counterpoints(
        self,
        pro_evidence: Optional[Evidence],
        con_evidence: Optional[Evidence],
        verdict: str
    ) -> List[str]:
        """
        Extract counterpoints to the main verdict.

        Args:
            pro_evidence: Supporting evidence
            con_evidence: Refuting evidence
            verdict: Final verdict type

        Returns:
            List of counterpoint statements
        """
        counterpoints = []

        if verdict == "Support":
            # Include notable concerns despite support
            if con_evidence and con_evidence.claims:
                # Take 1-2 most important concerns
                for claim in con_evidence.claims[:2]:
                    if con_evidence.confidence > 0.3:
                        counterpoints.append(claim)

                # Add qualifier if concerns are weak
                if con_evidence.confidence < 0.4:
                    counterpoints.append("Note: Contradicting evidence is limited")

        elif verdict == "Refute":
            # Include positive aspects despite refutation
            if pro_evidence and pro_evidence.claims:
                # Take 1-2 positive points
                for claim in pro_evidence.claims[:2]:
                    if pro_evidence.confidence > 0.3:
                        counterpoints.append(claim)

                # Add qualifier if positives are weak
                if pro_evidence.confidence < 0.4:
                    counterpoints.append("Note: Supporting evidence is limited")

        else:  # Inconclusive
            # Highlight strongest points from each side
            if pro_evidence and pro_evidence.confidence > 0.4:
                if len(pro_evidence.claims) > 1:
                    counterpoints.append(f"Positive: {pro_evidence.claims[1]}")
            if con_evidence and con_evidence.confidence > 0.4:
                if len(con_evidence.claims) > 1:
                    counterpoints.append(f"Concern: {con_evidence.claims[1]}")

        return counterpoints