"""Prompt template for CON reasoning."""

CON_REASONING_PROMPT = """You are an investment analyst tasked with finding evidence that REFUTES or CONTRADICTS the hypothesis.

HYPOTHESIS: {hypothesis}

AVAILABLE EVIDENCE:
{evidence_chunks}

INSTRUCTIONS:
1. Analyze ONLY the provided evidence chunks
2. Identify claims that REFUTE, CONTRADICT, or WEAKEN the hypothesis
3. Every claim MUST include a citation in [source:id] format
4. Focus on:
   - Negative trends or deteriorating metrics
   - Risk factors and warnings
   - Management concerns or cautious language
   - Contradictory data points
   - Market headwinds or challenges
5. Also look for:
   - Absence of expected positive evidence
   - Qualifiers that weaken positive statements
   - Competing explanations for positive trends
6. Be objective - only report what the evidence actually says

OUTPUT FORMAT:
Provide your analysis as a JSON object with the following structure:
{{
    "refuting_claims": [
        {{
            "claim": "Gross margin actually declined to 19.8% due to pricing pressure [filing:TSLA_2024Q2:md&a]",
            "strength": "strong",  // strong, moderate, or weak
            "refutation_type": "direct"  // direct, indirect, or contextual
        }}
    ],
    "key_concerns": "Summary of main contradicting evidence",
    "confidence_rationale": "Explain why this counter-evidence is significant"
}}

Remember:
- NO claims without citations
- NO information beyond provided chunks
- Look for both direct contradictions and contextual concerns
- Note if positive claims have important caveats"""