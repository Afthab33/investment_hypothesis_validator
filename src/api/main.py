"""
FastAPI backend for Investment Hypothesis Validator.
Provides REST API endpoint for validating investment hypotheses.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langchain_core.messages import HumanMessage
from src.graph.ihv_graph import create_ihv_graph

# Initialize FastAPI app
app = FastAPI(
    title="Investment Hypothesis Validator API",
    description="Validate investment hypotheses using AI-powered evidence analysis",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class ValidateRequest(BaseModel):
    query: str = Field(..., description="Investment hypothesis to validate", min_length=5)


class Citation(BaseModel):
    id: int
    raw: str
    display_text: str
    source_type: str  # "filing", "call", "chat"
    ticker: str
    company_name: str
    fiscal_period: Optional[str] = None
    date: Optional[str] = None
    speaker_role: Optional[str] = None
    section: Optional[str] = None
    full_text: Optional[str] = None


class EvidenceResponse(BaseModel):
    confidence: float
    claims: List[str]
    citations: List[Citation]


class VerdictResponse(BaseModel):
    verdict: str
    confidence: float
    rationale: List[str]
    counterpoints: Optional[List[str]] = None
    pro_evidence: Optional[EvidenceResponse] = None
    con_evidence: Optional[EvidenceResponse] = None
    tone_delta: Optional[str] = None
    timestamp: str
    execution_time_seconds: Optional[float] = None


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None


def parse_citation(citation_str: str, citation_id: int, chunks: List = None) -> Citation:
    """Parse citation string into structured Citation object."""
    import re

    # Default values
    source_type = "unknown"
    ticker = ""
    company_name = ""
    fiscal_period = None
    date = None
    speaker_role = None
    section = None
    full_text = None

    # Parse citation format: [source_type:TICKER_PERIOD:details]
    # Examples: [filing:TSLA_2024Q3:general], [call:TSLA_2024Q1:CFO], [chat:TSLA:2024-10-15:analyst]

    match = re.match(r'\[(\w+):([A-Z]+)(?:_(\w+))?(?::(.+?))?\]', citation_str)
    if match:
        source_type = match.group(1)  # filing, call, chat
        ticker = match.group(2)       # TSLA
        fiscal_period = match.group(3) # 2024Q3, 2024Q1, etc.
        details = match.group(4)      # general, CFO, date:role, etc.

        # Map ticker to company name (simple mapping, could be enhanced)
        company_map = {"TSLA": "Tesla", "AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Google"}
        company_name = company_map.get(ticker, ticker)

        # Parse details based on source type
        if source_type == "filing":
            section = details if details else "general"
            display_text = f"{company_name} {fiscal_period or ''} Filing - {section.replace('_', ' ').title()}"
        elif source_type == "call":
            speaker_role = details if details else "Speaker"
            display_text = f"{company_name} {fiscal_period or ''} Earnings Call - {speaker_role}"
        elif source_type == "chat":
            if details:
                parts = details.split(':')
                if len(parts) >= 1:
                    date = parts[0]
                if len(parts) >= 2:
                    speaker_role = parts[1]
            display_text = f"Trading Chat - {speaker_role or 'Trader'} ({date or 'Recent'})"
        else:
            display_text = citation_str
    else:
        display_text = citation_str

    # Try to find the full chunk text if chunks are provided
    if chunks:
        # Citation string might have brackets like [filing:TSLA_2024Q3:general]
        # But the evidence citations don't have brackets
        citation_to_match = citation_str.strip('[]')

        for chunk in chunks:
            if hasattr(chunk, 'get_citation'):
                chunk_citation = chunk.get_citation().strip('[]')
                if chunk_citation == citation_to_match:
                    full_text = chunk.text if hasattr(chunk, 'text') else None
                    break

    return Citation(
        id=citation_id,
        raw=citation_str,
        display_text=display_text,
        source_type=source_type,
        ticker=ticker,
        company_name=company_name,
        fiscal_period=fiscal_period,
        date=date,
        speaker_role=speaker_role,
        section=section,
        full_text=full_text
    )


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for Render."""
    return {
        "status": "healthy",
        "service": "Investment Hypothesis Validator API",
        "timestamp": datetime.utcnow().isoformat()
    }


# Main validation endpoint
@app.post("/validate", response_model=VerdictResponse)
async def validate_hypothesis(request: ValidateRequest):
    """
    Validate an investment hypothesis.

    Args:
        request: ValidateRequest with query string

    Returns:
        VerdictResponse with verdict, evidence, and citations

    Raises:
        HTTPException: If validation fails
    """
    try:
        # Create graph
        graph_app = create_ihv_graph()

        # Track execution time
        import time
        start_time = time.time()

        # Run validation
        result = graph_app.invoke({
            "messages": [HumanMessage(content=request.query)]
        })

        execution_time = time.time() - start_time

        # Extract verdict and chunks
        verdict = result.get("verdict")
        chunks = result.get("retrieved_chunks", [])

        if not verdict:
            raise HTTPException(
                status_code=500,
                detail="No verdict generated. Please try a different query."
            )

        # Parse citations for pro evidence
        pro_citations = []
        if verdict.pro_evidence and verdict.pro_evidence.citations:
            citation_id = 1
            for citation_str in verdict.pro_evidence.citations:
                pro_citations.append(parse_citation(citation_str, citation_id, chunks))
                citation_id += 1

        # Parse citations for con evidence
        con_citations = []
        if verdict.con_evidence and verdict.con_evidence.citations:
            citation_id = 1
            for citation_str in verdict.con_evidence.citations:
                con_citations.append(parse_citation(citation_str, citation_id, chunks))
                citation_id += 1

        # Build response
        response = VerdictResponse(
            verdict=verdict.verdict,
            confidence=verdict.confidence,
            rationale=verdict.rationale,
            counterpoints=verdict.counterpoints if verdict.counterpoints else None,
            pro_evidence=EvidenceResponse(
                confidence=verdict.pro_evidence.confidence,
                claims=verdict.pro_evidence.claims,
                citations=pro_citations
            ) if verdict.pro_evidence else None,
            con_evidence=EvidenceResponse(
                confidence=verdict.con_evidence.confidence,
                claims=verdict.con_evidence.claims,
                citations=con_citations
            ) if verdict.con_evidence else None,
            tone_delta=verdict.tone_delta if verdict.tone_delta else None,
            timestamp=datetime.utcnow().isoformat(),
            execution_time_seconds=round(execution_time, 2)
        )

        return response

    except Exception as e:
        # Log error (in production, use proper logging)
        print(f"Error validating hypothesis: {str(e)}")

        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "Investment Hypothesis Validator API",
        "version": "1.0.0",
        "endpoints": {
            "validate": "POST /validate - Validate an investment hypothesis",
            "health": "GET /health - Health check",
            "docs": "GET /docs - Interactive API documentation"
        },
        "status": "operational"
    }


if __name__ == "__main__":
    import uvicorn

    # Run server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True
    )
