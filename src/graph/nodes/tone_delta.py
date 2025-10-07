"""Tone delta analyzer for tracking sentiment shifts over time."""

from typing import Dict, Any, List
from collections import defaultdict
from src.graph.nodes.base import BaseNode
from src.graph.nodes.utils import get_state_attr
from src.retrieval.state import IHVState


class ToneDeltaAnalyzer(BaseNode):
    """Analyze sentiment shifts across time periods."""

    def __init__(self):
        super().__init__("tone_delta", temperature=0.0)

    def process(self, state: IHVState) -> Dict[str, Any]:
        """
        Analyze tone/sentiment changes over time.

        Args:
            state: Current state with verdict

        Returns:
            State updates with tone_delta in verdict
        """
        verdict = get_state_attr(state, 'verdict')
        if not verdict:
            return {}

        # Collect all chunks with sentiment data
        all_chunks = []
        if verdict.pro_evidence:
            all_chunks.extend(verdict.pro_evidence.chunks)
        if verdict.con_evidence:
            all_chunks.extend(verdict.con_evidence.chunks)

        if not all_chunks:
            return {}

        # Group by fiscal period and analyze sentiment
        tone_analysis = self._analyze_sentiment_by_period(all_chunks)

        if tone_analysis:
            # Update verdict with tone delta
            verdict.tone_delta = tone_analysis
            return {"verdict": verdict}

        return {}

    def _analyze_sentiment_by_period(self, chunks: List) -> str:
        """
        Analyze sentiment shifts across fiscal periods.

        Args:
            chunks: List of RetrievedChunk objects

        Returns:
            Tone delta description or None
        """
        # Group sentiments by period
        period_sentiments = defaultdict(list)

        for chunk in chunks:
            if chunk.fiscal_period and chunk.sentiment_score is not None:
                period_sentiments[chunk.fiscal_period].append({
                    'score': chunk.sentiment_score,
                    'sentiment': chunk.sentiment,
                    'source': chunk.source_type
                })

        if len(period_sentiments) < 2:
            return None  # Need at least 2 periods for comparison

        # Sort periods chronologically
        sorted_periods = sorted(period_sentiments.keys())

        # Calculate average sentiment per period
        period_averages = {}
        for period in sorted_periods:
            scores = [s['score'] for s in period_sentiments[period]]
            period_averages[period] = sum(scores) / len(scores)

        # Compare latest vs previous
        if len(sorted_periods) >= 2:
            latest = sorted_periods[-1]
            previous = sorted_periods[-2]

            latest_score = period_averages[latest]
            previous_score = period_averages[previous]

            delta = latest_score - previous_score

            # Determine significance
            if abs(delta) < 0.1:
                return None  # Not significant

            # Build analysis
            direction = "more positive" if delta > 0 else "more negative"
            magnitude = "significantly" if abs(delta) > 0.3 else "moderately"

            # Add context from sentiments
            latest_sentiments = period_sentiments[latest]
            dominant_sentiment = max(
                set(s['sentiment'] for s in latest_sentiments if s['sentiment']),
                key=lambda x: sum(1 for s in latest_sentiments if s.get('sentiment') == x),
                default=None
            )

            analysis = (
                f"Sentiment shifted {magnitude} {direction} from {previous} to {latest}. "
                f"Latest period shows {dominant_sentiment} tone "
                f"(avg score: {latest_score:+.2f} vs {previous_score:+.2f})."
            )

            return analysis

        return None