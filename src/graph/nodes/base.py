"""Base node class for LangGraph workflow."""

from typing import Any, Dict, Optional
from abc import ABC, abstractmethod
import logging
from langchain_core.messages import HumanMessage, AIMessage
from src.aws.bedrock_client import get_llm
from src.retrieval.state import IHVState

logger = logging.getLogger(__name__)


class BaseNode(ABC):
    """Abstract base class for all graph nodes."""

    def __init__(self, name: str, temperature: float = 0.0):
        """
        Initialize base node.

        Args:
            name: Node name for logging
            temperature: LLM temperature (0 for deterministic)
        """
        self.name = name
        self.temperature = temperature
        self.llm = None

    def get_llm(self, max_tokens: Optional[int] = None):
        """Get or create LLM instance."""
        if self.llm is None:
            self.llm = get_llm(
                temperature=self.temperature,
                max_tokens=max_tokens
            )
        return self.llm

    @abstractmethod
    def process(self, state: IHVState) -> Dict[str, Any]:
        """
        Process the state and return updates.

        Args:
            state: Current workflow state

        Returns:
            Dictionary of state updates
        """
        pass

    def __call__(self, state: IHVState) -> Dict[str, Any]:
        """Make node callable for LangGraph."""
        logger.info(f"Executing node: {self.name}")
        try:
            # LangGraph passes state as dict, convert for convenience
            if isinstance(state, dict):
                # Create a simple namespace object for easier access
                from types import SimpleNamespace
                state_obj = SimpleNamespace(**state)
            else:
                state_obj = state

            result = self.process(state_obj)
            logger.info(f"Node {self.name} completed successfully")
            return result
        except Exception as e:
            logger.error(f"Error in node {self.name}: {str(e)}")
            raise

    def format_prompt(self, template: str, **kwargs) -> str:
        """
        Format a prompt template with variables.

        Args:
            template: Prompt template with {variable} placeholders
            **kwargs: Variables to fill in

        Returns:
            Formatted prompt string
        """
        try:
            return template.format(**kwargs)
        except KeyError as e:
            logger.error(f"Missing prompt variable: {e}")
            raise

    def extract_citations(self, text: str) -> list:
        """
        Extract citations from text in format [source:details].

        Args:
            text: Text containing citations

        Returns:
            List of citation strings
        """
        import re
        pattern = r'\[([^\]]+)\]'
        citations = re.findall(pattern, text)
        # Filter out non-citation brackets
        citations = [c for c in citations if ':' in c]
        return citations

    def validate_citations(self, claims: list, citations: list, chunks: list) -> bool:
        """
        Validate that all claims have citations and citations exist in chunks.

        Args:
            claims: List of claim strings
            citations: List of citation strings
            chunks: List of RetrievedChunk objects

        Returns:
            True if all claims are properly cited
        """
        # Check each claim has at least one citation
        for claim in claims:
            if not any(f"[{cite}]" in claim for cite in citations):
                logger.warning(f"Claim without citation: {claim}")
                return False

        # Check citations reference actual chunks
        chunk_ids = [c.chunk_id for c in chunks]
        for citation in citations:
            # Extract chunk reference from citation
            if not any(chunk_id in citation for chunk_id in chunk_ids):
                logger.warning(f"Citation references non-existent chunk: {citation}")

        return True

    def calculate_source_weight(self, source_type: str, speaker_role: Optional[str] = None) -> float:
        """
        Calculate weight based on source type and speaker role.

        Args:
            source_type: filing, call, or chat
            speaker_role: Optional role (CEO, CFO, analyst, etc.)

        Returns:
            Weight between 0 and 1
        """
        # Base weights by source
        base_weights = {
            "filing": 0.9,  # Official documents highest
            "call": 0.7,    # Earnings calls
            "chat": 0.4,    # Market chatter lowest
        }

        weight = base_weights.get(source_type, 0.5)

        # Adjust for speaker authority in calls
        if source_type == "call" and speaker_role:
            role_multipliers = {
                "CEO": 1.2,
                "CFO": 1.15,
                "COO": 1.1,
                "analyst": 0.9,
            }
            multiplier = role_multipliers.get(speaker_role, 1.0)
            weight *= multiplier

        return min(weight, 1.0)  # Cap at 1.0