"""Report formatter node for final output."""

import json
from typing import Dict, Any, List
from datetime import datetime
from src.graph.nodes.base import BaseNode
from src.graph.nodes.utils import get_state_attr
from src.retrieval.state import IHVState, Verdict


class ReportFormatter(BaseNode):
    """Format final verdict into structured report."""

    def __init__(self, output_format: str = "both"):
        """
        Initialize report formatter.

        Args:
            output_format: "json", "markdown", or "both"
        """
        super().__init__("report_formatter", temperature=0.0)
        self.output_format = output_format

    def process(self, state: IHVState) -> Dict[str, Any]:
        """
        Format verdict into final report.

        Args:
            state: Current state with verdict

        Returns:
            Formatted report (added to messages)
        """
        verdict = get_state_attr(state, 'verdict')
        if not verdict:
            return {}

        query = get_state_attr(state, 'normalized_query')

        # Generate both formats
        json_report = self._format_json(verdict, query)
        markdown_report = self._format_markdown(verdict, query)

        # Prepare final message
        if self.output_format == "json":
            final_output = json.dumps(json_report, indent=2)
        elif self.output_format == "markdown":
            final_output = markdown_report
        else:  # both
            final_output = f"{markdown_report}\n\n---\nJSON Output:\n```json\n{json.dumps(json_report, indent=2)}\n```"

        # Add to messages as AI response
        from langchain_core.messages import AIMessage
        return {
            "messages": [AIMessage(content=final_output)]
        }

    def _format_json(self, verdict: Verdict, query) -> dict:
        """Format as JSON report."""
        report = {
            "query": query.original_query if query else "N/A",
            "verdict": verdict.verdict,
            "confidence": round(verdict.confidence, 3),
            "timestamp": datetime.now().isoformat(),
            "analysis": {
                "rationale": verdict.rationale,
                "counterpoints": verdict.counterpoints,
            },
            "evidence_summary": {
                "supporting": {
                    "confidence": round(verdict.pro_evidence.confidence, 3) if verdict.pro_evidence else 0,
                    "claim_count": len(verdict.pro_evidence.claims) if verdict.pro_evidence else 0,
                    "top_claims": verdict.pro_evidence.claims[:3] if verdict.pro_evidence else []
                },
                "refuting": {
                    "confidence": round(verdict.con_evidence.confidence, 3) if verdict.con_evidence else 0,
                    "claim_count": len(verdict.con_evidence.claims) if verdict.con_evidence else 0,
                    "top_claims": verdict.con_evidence.claims[:3] if verdict.con_evidence else []
                }
            },
            "metadata": {
                "ticker": query.ticker if query else None,
                "fiscal_period": query.fiscal_period if query else None,
                "sources_analyzed": self._count_sources(verdict),
            }
        }

        if verdict.tone_delta:
            report["tone_delta"] = verdict.tone_delta

        return report

    def _format_markdown(self, verdict: Verdict, query) -> str:
        """Format as markdown report."""
        lines = []

        # Header
        lines.append("# Investment Hypothesis Validation Report")
        lines.append("")
        lines.append(f"**Query:** {query.original_query if query else 'N/A'}")
        if query and query.ticker:
            lines.append(f"**Ticker:** {query.ticker}")
        if query and query.fiscal_period:
            lines.append(f"**Period:** {query.fiscal_period}")
        lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # Verdict Box
        lines.append("## 📊 Verdict")
        lines.append("")
        emoji = self._get_verdict_emoji(verdict.verdict)
        lines.append(f"### {emoji} **{verdict.verdict}**")
        lines.append(f"**Confidence:** {verdict.confidence:.1%}")
        lines.append("")

        # Main Rationale
        lines.append("## 📝 Key Findings")
        lines.append("")
        for i, point in enumerate(verdict.rationale, 1):
            lines.append(f"{i}. {point}")
        lines.append("")

        # Counterpoints
        if verdict.counterpoints:
            lines.append("## ⚠️ Important Considerations")
            lines.append("")
            for point in verdict.counterpoints:
                lines.append(f"- {point}")
            lines.append("")

        # Evidence Summary
        lines.append("## 🔍 Evidence Analysis")
        lines.append("")

        if verdict.pro_evidence:
            lines.append(f"### Supporting Evidence (Confidence: {verdict.pro_evidence.confidence:.1%})")
            lines.append(f"- {len(verdict.pro_evidence.claims)} supporting claims found")
            lines.append(f"- {len(verdict.pro_evidence.chunks)} relevant chunks analyzed")
            lines.append("")

        if verdict.con_evidence:
            lines.append(f"### Contradicting Evidence (Confidence: {verdict.con_evidence.confidence:.1%})")
            lines.append(f"- {len(verdict.con_evidence.claims)} refuting claims found")
            lines.append(f"- {len(verdict.con_evidence.chunks)} relevant chunks analyzed")
            lines.append("")

        # Tone Delta
        if verdict.tone_delta:
            lines.append("## 🎭 Sentiment Shift")
            lines.append("")
            lines.append(verdict.tone_delta)
            lines.append("")

        # Sources
        lines.append("## 📚 Sources Analyzed")
        lines.append("")
        sources = self._count_sources(verdict)
        for source_type, count in sources.items():
            lines.append(f"- {source_type.title()}: {count} chunks")
        lines.append("")

        # Footer
        lines.append("---")
        lines.append("*Generated by Investment Hypothesis Validator*")

        return "\n".join(lines)

    def _get_verdict_emoji(self, verdict: str) -> str:
        """Get emoji for verdict type."""
        emojis = {
            "Support": "✅",
            "Refute": "❌",
            "Inconclusive": "❔"
        }
        return emojis.get(verdict, "📊")

    def _count_sources(self, verdict: Verdict) -> dict:
        """Count chunks by source type."""
        source_counts = {}

        all_chunks = []
        if verdict.pro_evidence:
            all_chunks.extend(verdict.pro_evidence.chunks)
        if verdict.con_evidence:
            all_chunks.extend(verdict.con_evidence.chunks)

        # Count unique chunks by source
        seen_ids = set()
        for chunk in all_chunks:
            if chunk.chunk_id not in seen_ids:
                seen_ids.add(chunk.chunk_id)
                source_type = chunk.source_type
                source_counts[source_type] = source_counts.get(source_type, 0) + 1

        return source_counts