"""Question normalization node for extracting entities and expanding queries."""

from typing import Dict, Any, List, Optional
import re
from langchain_core.messages import HumanMessage
from src.graph.nodes.base import BaseNode
from src.retrieval.state import IHVState, NormalizedQuery


class QuestionNormalizer(BaseNode):
    """Extract entities and normalize the user question."""

    def __init__(self):
        super().__init__("question_normalizer", temperature=0.0)

        # Ticker mappings
        self.ticker_map = {
            "tesla": "TSLA",
            "nvidia": "NVDA",
            "apple": "AAPL",
            "microsoft": "MSFT",
            "amazon": "AMZN",
            "google": "GOOGL",
            "meta": "META",
            "netflix": "NFLX",
        }

        # Metric synonyms
        self.metric_synonyms = {
            "gross margin": ["gm", "gross profit margin", "gross-margin"],
            "operating margin": ["opex", "operating profit margin", "op margin"],
            "revenue": ["sales", "top line", "revenue growth"],
            "earnings": ["eps", "net income", "bottom line", "profit"],
            "guidance": ["outlook", "forecast", "forward guidance"],
            "margin pressure": ["margin compression", "margin headwinds"],
            "ai": ["artificial intelligence", "machine learning", "ml"],
        }

        # Quarter patterns
        self.quarter_patterns = {
            r"Q1\s*202\d": "Q1",
            r"Q2\s*202\d": "Q2",
            r"Q3\s*202\d": "Q3",
            r"Q4\s*202\d": "Q4",
            r"first quarter": "Q1",
            r"second quarter": "Q2",
            r"third quarter": "Q3",
            r"fourth quarter": "Q4",
        }

    def process(self, state: IHVState) -> Dict[str, Any]:
        """
        Normalize the question and extract entities.

        Args:
            state: Current workflow state with messages

        Returns:
            State updates with normalized_query
        """
        # Handle state as dict (LangGraph passes dict)
        if isinstance(state, dict):
            messages = state.get("messages", [])
        else:
            messages = state.messages

        # Get the user question from messages
        user_question = ""
        for message in messages:
            if isinstance(message, HumanMessage):
                user_question = message.content
                break

        if not user_question:
            return {"normalized_query": None}

        # Extract entities
        ticker = self._extract_ticker(user_question)
        company = self._extract_company(user_question)
        fiscal_period = self._extract_fiscal_period(user_question)
        metrics = self._extract_metrics(user_question)
        keywords = self._expand_keywords(user_question, metrics)

        # Create normalized query
        normalized = user_question.lower()

        # Expand abbreviations
        for metric, synonyms in self.metric_synonyms.items():
            for syn in synonyms:
                if syn in normalized and metric not in normalized:
                    normalized = normalized.replace(syn, metric)

        # Build filters
        filters = {}
        if ticker:
            filters["ticker"] = ticker
        if fiscal_period:
            filters["fiscal_period"] = fiscal_period

        # Create NormalizedQuery object
        normalized_query = NormalizedQuery(
            original_query=user_question,
            normalized_query=normalized,
            ticker=ticker,
            company=company,
            fiscal_period=fiscal_period,
            metrics=metrics,
            keywords=keywords,
            filters=filters
        )

        return {"normalized_query": normalized_query}

    def _extract_ticker(self, text: str) -> Optional[str]:
        """Extract ticker symbol from text."""
        text_lower = text.lower()

        # Check for explicit tickers (e.g., TSLA, $TSLA)
        ticker_pattern = r'\$?([A-Z]{2,5})\b'
        matches = re.findall(ticker_pattern, text)
        if matches:
            return matches[0]

        # Check company name mappings
        for company, ticker in self.ticker_map.items():
            if company in text_lower:
                return ticker

        return None

    def _extract_company(self, text: str) -> Optional[str]:
        """Extract company name from text."""
        text_lower = text.lower()

        # Company name patterns
        companies = {
            "tesla": "Tesla, Inc.",
            "nvidia": "NVIDIA Corporation",
            "apple": "Apple Inc.",
            "microsoft": "Microsoft Corporation",
            "amazon": "Amazon.com, Inc.",
            "google": "Alphabet Inc.",
            "meta": "Meta Platforms, Inc.",
            "netflix": "Netflix, Inc.",
        }

        for key, name in companies.items():
            if key in text_lower:
                return name

        return None

    def _extract_fiscal_period(self, text: str) -> Optional[str]:
        """Extract fiscal period from text."""
        import re

        # Look for year
        year_match = re.search(r'202[3-5]', text)
        year = year_match.group(0) if year_match else None

        # Look for quarter
        quarter = None
        for pattern, q in self.quarter_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                quarter = q
                break

        if year and quarter:
            return f"{year}{quarter}"
        elif year:
            return year
        elif quarter:
            # Default to current/recent year if only quarter specified
            return f"2024{quarter}"

        return None

    def _extract_metrics(self, text: str) -> List[str]:
        """Extract financial metrics mentioned in the query."""
        text_lower = text.lower()
        found_metrics = []

        # Check for each metric and its synonyms
        for metric, synonyms in self.metric_synonyms.items():
            if metric in text_lower:
                found_metrics.append(metric)
            else:
                for syn in synonyms:
                    if syn in text_lower:
                        found_metrics.append(metric)
                        break

        # Additional metric patterns
        metric_patterns = [
            (r'margin[s]?\s+(?:pressure|compression|expansion)', 'margin pressure'),
            (r'(?:revenue|sales)\s+growth', 'revenue growth'),
            (r'(?:earnings|eps)\s+(?:beat|miss)', 'earnings performance'),
            (r'cash\s+flow', 'cash flow'),
            (r'(?:capex|capital\s+expenditure)', 'capex'),
        ]

        for pattern, metric_name in metric_patterns:
            if re.search(pattern, text_lower):
                if metric_name not in found_metrics:
                    found_metrics.append(metric_name)

        return found_metrics

    def _expand_keywords(self, text: str, metrics: List[str]) -> List[str]:
        """Expand keywords for better retrieval."""
        keywords = []
        text_lower = text.lower()

        # Add metric-related keywords
        for metric in metrics:
            keywords.append(metric)
            # Add related terms
            if "margin" in metric:
                keywords.extend(["profitability", "costs", "pricing"])
            elif "revenue" in metric:
                keywords.extend(["sales", "growth", "demand"])
            elif "guidance" in metric:
                keywords.extend(["outlook", "forecast", "expectations"])

        # Add sentiment keywords if present
        sentiment_keywords = {
            "improving": ["improvement", "better", "increase", "growth"],
            "declining": ["decline", "decrease", "pressure", "headwind"],
            "stable": ["stability", "steady", "unchanged"],
            "easing": ["ease", "relief", "improvement", "recovery"],
        }

        for key, related in sentiment_keywords.items():
            if key in text_lower:
                keywords.extend(related)

        # Remove duplicates
        return list(set(keywords))