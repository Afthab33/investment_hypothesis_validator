"""
State definitions for LangGraph workflow.
Uses MessagesState pattern from latest LangGraph.
"""

from typing import TypedDict, List, Optional, Literal
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    """A single retrieved chunk with metadata matching OpenSearch schema."""

    # Core identifiers (all sources)
    chunk_id: str
    parent_doc_id: str
    text: str
    source_type: Literal["filing", "call", "chat"]
    ticker: str
    company_name: str

    # Temporal fields (all sources)
    fiscal_year: Optional[int] = None
    fiscal_quarter: Optional[str] = None
    fiscal_period: Optional[str] = None

    # Date fields (source-specific)
    filing_date: Optional[str] = None  # For filings
    call_date: Optional[str] = None    # For earnings calls
    timestamp: Optional[str] = None    # For chats

    # Scores
    vector_score: float = 0.0
    keyword_score: float = 0.0
    hybrid_score: float = 0.0
    recency_score: float = 1.0
    final_score: float = 0.0

    # Filing-specific metadata
    filing_type: Optional[str] = None  # 10-K, 10-Q
    section: Optional[str] = None      # risk_factors, md&a, etc.
    has_tables: Optional[bool] = None

    # Call-specific metadata
    speaker: Optional[str] = None      # Speaker name
    speaker_role: Optional[str] = None # CEO, CFO, Analyst, etc.
    is_company_speaker: Optional[bool] = None

    # Chat-specific metadata
    trader_id: Optional[str] = None
    trader_role: Optional[str] = None  # portfolio_manager, analyst, trader
    message_index: Optional[int] = None

    # Enrichment fields (all sources)
    contains_numbers: Optional[bool] = None
    forward_looking: Optional[bool] = None
    importance_score: Optional[float] = None

    # LLM-enhanced fields (calls and chats)
    sentiment: Optional[str] = None       # bullish, neutral, bearish
    sentiment_score: Optional[float] = None  # -1.0 to 1.0
    confidence_level: Optional[float] = None  # 0.0 to 1.0 (calls)

    # Chat-specific signals
    signal_type: Optional[str] = None     # news, analysis, rumor, recommendation, general
    credibility_score: Optional[float] = None  # 0.0 to 1.0
    actionability: Optional[str] = None   # high, medium, low
    urgency: Optional[str] = None         # immediate, short_term, long_term

    @property
    def document_date(self) -> Optional[str]:
        """Get the appropriate date field based on source type."""
        if self.source_type == "filing":
            return self.filing_date
        elif self.source_type == "call":
            return self.call_date
        elif self.source_type == "chat":
            return self.timestamp
        return None
    
    def get_citation(self) -> str:
        """Generate a citation string for this chunk."""
        if self.source_type == "filing":
            section_info = f":{self.section}" if self.section else ""
            period_info = f"_{self.fiscal_period}" if self.fiscal_period else ""
            return f"[filing:{self.ticker}{period_info}{section_info}]"
        elif self.source_type == "call":
            speaker_info = f":{self.speaker_role}" if self.speaker_role else ""
            period_info = f"_{self.fiscal_period}" if self.fiscal_period else ""
            return f"[call:{self.ticker}{period_info}{speaker_info}]"
        else:  # chat
            date_str = self.timestamp[:10] if self.timestamp else "unknown"
            trader_info = f":{self.trader_role}" if self.trader_role else ""
            return f"[chat:{self.ticker}:{date_str}{trader_info}]"


class NormalizedQuery(BaseModel):
    """Normalized and enriched query."""
    
    original_query: str
    normalized_query: str
    ticker: Optional[str] = None
    company: Optional[str] = None
    fiscal_period: Optional[str] = None
    metrics: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    filters: dict = Field(default_factory=dict)


class Evidence(BaseModel):
    """Evidence for PRO or CON stance."""
    
    stance: Literal["pro", "con"]
    claims: List[str]
    citations: List[str]
    chunks: List[RetrievedChunk]
    confidence: float = 0.0


class Verdict(BaseModel):
    """Final verdict with confidence and evidence."""
    
    verdict: Literal["Support", "Refute", "Inconclusive"]
    confidence: float
    rationale: List[str]
    counterpoints: List[str]
    tone_delta: Optional[str] = None
    pro_evidence: Optional[Evidence] = None
    con_evidence: Optional[Evidence] = None


class IHVState(MessagesState):
    """
    State for Investment Hypothesis Validator workflow.
    Extends MessagesState with custom fields.
    """
    
    # Query processing
    normalized_query: Optional[NormalizedQuery] = None
    
    # Retrieval
    retrieved_chunks: List[RetrievedChunk] = Field(default_factory=list)
    retrieval_quality: float = 0.0
    needs_rewrite: bool = False
    _rewrite_attempted: bool = False  # Internal flag to prevent rewrite loops
    
    # Reasoning
    pro_evidence: Optional[Evidence] = None
    con_evidence: Optional[Evidence] = None
    
    # Final output
    verdict: Optional[Verdict] = None
