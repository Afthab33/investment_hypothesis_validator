# Investment Hypothesis Validator (IHV)

**An AI-orchestrated workflow for evidence-based investment decisions**

## Project Overview

The Investment Hypothesis Validator is an AI-orchestrated workflow where a PM/analyst asks nuanced questions (e.g., "Is Tesla's gross margin improving?"). The system retrieves evidence from earnings calls, 10-Q/10-K filings, and trading chat, then returns a verdict (Support / Refute / Inconclusive) with confidence scores, cited snippets, counterpoints, and tone-delta analysis.

**Primary Outcome:** A decision-ready, evidence-bound insight in ~20-25 seconds, using LangGraph + LangChain + AWS Bedrock + OpenSearch Serverless.

## Architecture

```
[Raw Docs] -> [Ingest/Parse] -> [Chunk+Enrich] -> [Embeddings] -> [OpenSearch Serverless]
                                                             |
User Question -> [QuestionNormalize] -> [StratifiedRetriever BM25+Vector] -> [RerankDiversify]
                -> (low quality?) -> [QueryRewrite] ---------------------------^
                                   |
                                   v
                         [ProReasoner]  -->  [ConReasoner]
                                   \            /
                                    \          /
                                 [VerdictSynthesizer]
                                         |
                                  [ToneDeltaAnalyzer]
                                         |
                                  [ReportFormatter]
                                         |
                                   [JSON + Markdown]
```

## Core Features

### 1. Data Sources
- **Earnings call transcripts** - Management commentary and Q&A
- **10-Q/10-K filings** - Official financial statements and disclosures
- **Trading chat** - Real-time analyst and trader discussions

### 2. Retrieval System
- **Stratified hybrid retrieval** - Separate BM25 + kNN search per source type
- **Reciprocal Rank Fusion** - Intelligent merging of ranked results
- **Source diversity** - Ensures representation from filings, calls, and chat
- **Recency weighting** - Recent documents scored higher

### 3. Reasoning Layer (LangGraph)
- **QuestionNormalizer** - Extracts ticker, period, expands keywords
- **StratifiedRetriever** - Multi-source hybrid search
- **RerankDiversify** - Filters low-quality chunks, enforces diversity
- **QueryRewriter** - Fallback for low retrieval quality
- **ProReasoner** - Finds supporting evidence with citations
- **ConReasoner** - Finds refuting evidence with citations
- **VerdictSynthesizer** - Determines final verdict with confidence
- **ToneDeltaAnalyzer** - Tracks sentiment shifts across periods
- **ReportFormatter** - Generates structured output

### 4. Evidence-Bound Design
- **Every claim requires citation** - No hallucination beyond context
- **Citation format:**
  - Filings: `[filing:TSLA_2024Q4:section]`
  - Calls: `[call:TSLA_2024Q1:CFO]`
  - Chat: `[chat:TSLA:2024-07-20:trader]`
- **Abstains when insufficient evidence** - Returns "Inconclusive" rather than guessing

### 5. Dual-Stance Reasoning
- Separate PRO and CON analysis prevents confirmation bias
- Balanced evidence presentation
- Identifies contradictions in the data

### 6. Confidence Scoring
Heuristic combining:
- PRO vs CON evidence strength
- Source authority (filing > call > chat)
- Data type (quantitative > qualitative)
- Recency and consistency

## Repository Structure

```
investment_hypothesis_validator/
  README.md
  requirements.txt
  .env
  src/
    aws/
      bedrock_client.py          # LLM client with retry logic
      opensearch_client.py       # OpenSearch Serverless client
    retrieval/
      state.py                   # Pydantic state models
      hybrid_retriever.py        # Stratified retrieval implementation
    graph/
      ihv_graph.py              # LangGraph workflow
      nodes/
        base.py                 # Base node class
        question_normalize.py   # Query normalization
        retrieval_node.py       # Retrieval wrapper
        rerank_diversify.py     # Reranking and diversity
        query_rewrite.py        # Query expansion
        pro_reasoner.py         # Supporting evidence
        con_reasoner.py         # Refuting evidence
        verdict_synthesizer.py  # Final verdict
        tone_delta.py           # Sentiment analysis
        report_formatter.py     # Output generation
        utils.py                # Helper functions
    ingestion/
      ingest_fixed.py           # Document ingestion
  scripts/
    test_retrieval.py           # Test retrieval layer
    test_single_query.py        # Test complete workflow
    test_simple_workflow.py     # Debug workflow execution
    check_bedrock_quotas.py     # Check AWS quotas
  docs/
    UNIFIED_SCHEMA_REFERENCE.md           # Document schema
    OPENSEARCH_SCHEMA_AND_RETRIEVAL_GUIDE.md  # Retrieval guide
    IMPLEMENTATION_SUMMARY.md             # System overview
    CITATION_VERIFICATION.md              # Citation system docs
    AWS_BEDROCK_RATE_LIMITS_GUIDE.md     # Rate limit troubleshooting
    NEXT_STEPS.md                         # Implementation status
  data/
    raw/              # Raw source documents
    processed/        # Chunked documents ready for ingestion
```

## Quick Start

### Prerequisites

1. **AWS Account** with:
   - OpenSearch Serverless collection
   - Bedrock model access (Claude Sonnet 4 or Claude 3.5 Haiku)
   - IAM credentials with appropriate permissions

2. **Python 3.10+**

### Installation

```bash
# Clone repository
git clone <repo-url>
cd investment_hypothesis_validator

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your AWS credentials
```

### Environment Variables

```bash
# AWS Configuration
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key

# OpenSearch Serverless
OPENSEARCH_URL=https://your-collection.us-east-1.aoss.amazonaws.com
OPENSEARCH_INDEX_NAME=investment-documents

# Bedrock Configuration
BEDROCK_REGION=us-east-1
BEDROCK_EMBEDDING_MODEL=amazon.titan-embed-text-v2:0
BEDROCK_LLM_MODEL=us.anthropic.claude-sonnet-4-20250514-v1:0

# Retrieval Parameters
TOP_K_PER_SOURCE=5
VECTOR_WEIGHT=0.6
KEYWORD_WEIGHT=0.4
RECENCY_HALFLIFE_DAYS=14
```

### Running the System

```bash
# Test retrieval layer
python scripts/test_retrieval.py "Is Tesla's gross margin improving?"

# Run complete workflow
python scripts/test_single_query.py

# Debug workflow execution
python scripts/test_simple_workflow.py
```

## OpenSearch Schema

### Actual Schema Used (Important!)

The system uses a **nested metadata structure**:

```json
{
  "text": "chunk content here",
  "vector_field": [0.123, 0.456, ...],
  "metadata": {
    "chunk_id": "unique_id",
    "parent_doc_id": "doc_id",
    "source_type": "filing",
    "ticker": "TSLA",
    "company_name": "Tesla",
    "fiscal_period": "2024Q4",
    "filing_date": "2024-12-31",
    "section": "business",
    "timestamp": "2024-12-31T00:00:00"
  }
}
```

**Key Points:**
- All metadata fields are under `metadata.` prefix
- Vector field is named `vector_field` (not `embedding`)
- Text field searches require `.keyword` suffix for exact matches
- OpenSearch Serverless doesn't support `info()` or `stats()` endpoints

See [OPENSEARCH_SCHEMA_AND_RETRIEVAL_GUIDE.md](docs/OPENSEARCH_SCHEMA_AND_RETRIEVAL_GUIDE.md) for complete details.

## Example Output

**Query:** "Is Tesla's gross margin improving?"

**System Response:**

```
VERDICT: Inconclusive
CONFIDENCE: 28.3%
TIME: 23.6s

✅ SUPPORTING EVIDENCE (Confidence: 56.6%)

Claims with Citations:

1. Cost of automotive sales revenue decreased $2.32 billion, or 5%, in the nine
   months ended September 30, 2024 compared to 2023, primarily from lower raw
   material costs, freight and duties [filing:TSLA_2024Q3:general]

2. Cost per unit down 8% quarter-over-quarter, with expectations that margins
   should stabilize [chat:TSLA:2024-07-20:trader]

3. Production efficiency improving across all factories
   [chat:TSLA:2024-07-20:portfolio_manager]

❌ REFUTING EVIDENCE (Confidence: 45.9%)

Claims with Citations:

1. Auto margins declined from 18.9% to 18.5% in Q1 2024
   [call:TSLA_2024Q1:CFO]

2. Management acknowledged that pricing actions negatively impacted margins,
   which were only offset by cost reductions [call:TSLA_2024Q1:CFO]

3. Trader commentary suggests that when stripping out certain factors,
   'margins are ugly' [chat:TSLA:2024-07-20:trader]

📝 RATIONALE

1. Supporting: Cost of automotive sales revenue decreased $2.32 billion (5%),
   indicating potential margin improvement through reduced per-unit costs

2. However: Auto margins declined from 18.9% to 18.5% in Q1 2024, indicating
   margin compression

3. Mixed signals prevent definitive conclusion
```

## AWS Bedrock Model Recommendations

### Recommended: Claude Sonnet 4 (Cross-Region)
```bash
BEDROCK_LLM_MODEL=us.anthropic.claude-sonnet-4-20250514-v1:0
```
- Best reasoning quality
- Most detailed claim extraction
- Better citation accuracy
- **Quota:** 2 RPM (request quota increase to 200+ RPM)

### Alternative: Claude 3.5 Haiku (Higher Throughput)
```bash
BEDROCK_LLM_MODEL=us.anthropic.claude-3-5-haiku-20241022-v1:0
```
- Good reasoning quality
- Faster execution
- **Quota:** 20 RPM (better for development/testing)

### Rate Limit Considerations

Default Bedrock quotas are **very low** (1-2 RPM for Sonnet models). See [AWS_BEDROCK_RATE_LIMITS_GUIDE.md](docs/AWS_BEDROCK_RATE_LIMITS_GUIDE.md) for:
- How to check your current quotas
- Steps to request quota increases
- Alternative models with higher quotas
- Troubleshooting throttling errors

## System Performance

With adequate Bedrock quotas (50+ RPM):

| Metric | Value |
|--------|-------|
| Query Latency | 20-25 seconds |
| Throughput | 120-180 queries/hour |
| Citation Coverage | 100% (enforced) |
| Retrieval Precision | High (stratified + diverse) |
| Evidence Quality | Source-weighted and verified |

## Key Design Decisions

### 1. Score Threshold (Updated)
- Default `min_score_threshold = 0.01` in rerank node
- Original 0.3 was too high for actual score distributions
- Actual scores typically range 0.01 - 0.20

### 2. Sequential PRO/CON Execution
- Changed from parallel to sequential to avoid Bedrock throttling
- PRO reasoner completes before CON reasoner starts
- Adds ~2-4s latency but prevents rate limit errors

### 3. Retry Logic
- Exponential backoff: 1s → 2s → 4s → 8s → 16s
- Rate limiting: 2s minimum between LLM calls
- Handles transient Bedrock throttling gracefully

### 4. Query Rewrite Prevention
- `_rewrite_attempted` flag prevents infinite loops
- Rewrite only triggers once per query
- Quality threshold: 0.4 (triggers rewrite if below)

## Testing & Validation

### Retrieval Tests
```bash
python scripts/test_retrieval.py "query here"
```
Verifies:
- OpenSearch connection
- Hybrid search functionality
- Source diversity
- Metadata extraction

### End-to-End Tests
```bash
python scripts/test_single_query.py
```
Verifies:
- Complete workflow execution
- Citation quality
- PRO/CON reasoning
- Verdict synthesis
- Report formatting

### Debugging
```bash
python scripts/test_simple_workflow.py
```
Shows:
- Step-by-step node execution
- State updates at each stage
- Retrieval quality scores
- Evidence generation

## Production Readiness Checklist

- ✅ Core retrieval implemented
- ✅ Dual-stance reasoning working
- ✅ Citation system verified
- ✅ Error handling and retries
- ✅ Rate limiting
- ✅ State management
- ✅ Comprehensive documentation
- ⚠️ **Bedrock quotas** - Requires increase for production
- ⏳ Performance benchmarking at scale
- ⏳ Optional: Web UI (Streamlit/Gradio)
- ⏳ Optional: Batch processing
- ⏳ Optional: Caching layer

## Future Enhancements

### Phase 2
- **Request caching** - Cache LLM responses for identical queries
- **Batch processing** - Process multiple queries in parallel
- **Web interface** - Streamlit or Gradio UI
- **Export options** - PDF, Excel, PowerPoint

### Phase 3
- **Multi-company comparison** - Compare competitors side-by-side
- **Alert system** - Monitor specific hypotheses over time
- **REST API** - External integrations
- **Advanced analytics** - Trend tracking and aggregation

## Troubleshooting

### Common Issues

1. **No chunks returned**
   - Check `min_score_threshold` in `rerank_diversify.py`
   - Verify OpenSearch index has documents
   - Check metadata field names match schema

2. **Bedrock throttling**
   - Increase delay between calls in `bedrock_client.py`
   - Request quota increase via AWS Service Quotas
   - Consider switching to higher-quota model

3. **Infinite loop in graph**
   - Verify `_rewrite_attempted` flag is set
   - Check conditional edges in `ihv_graph.py`
   - Review query rewrite logic

4. **Missing citations**
   - Check evidence parsing in `base.py`
   - Verify LLM prompt enforces citations
   - Review regex pattern for citation extraction

See [docs/](docs/) for detailed troubleshooting guides.

## Documentation

- [UNIFIED_SCHEMA_REFERENCE.md](docs/UNIFIED_SCHEMA_REFERENCE.md) - Document metadata schema
- [OPENSEARCH_SCHEMA_AND_RETRIEVAL_GUIDE.md](docs/OPENSEARCH_SCHEMA_AND_RETRIEVAL_GUIDE.md) - OpenSearch implementation
- [IMPLEMENTATION_SUMMARY.md](docs/IMPLEMENTATION_SUMMARY.md) - System architecture
- [CITATION_VERIFICATION.md](docs/CITATION_VERIFICATION.md) - Citation system details
- [AWS_BEDROCK_RATE_LIMITS_GUIDE.md](docs/AWS_BEDROCK_RATE_LIMITS_GUIDE.md) - Rate limit management
- [NEXT_STEPS.md](docs/NEXT_STEPS.md) - Current status and roadmap

## License

MIT License

## Contact

For questions or issues, please open a GitHub issue or contact the maintainers.

---

**Built with:** LangGraph • LangChain • AWS Bedrock • OpenSearch Serverless • Claude
