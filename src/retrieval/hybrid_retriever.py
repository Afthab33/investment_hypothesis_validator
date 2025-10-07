"""
Stratified hybrid retrieval combining BM25 and vector search per source type.
Based on latest LangChain patterns.
"""

import os
from typing import List, Dict, Tuple, Optional, Literal
from datetime import datetime
import math

from langchain_core.documents import Document
from dotenv import load_dotenv

from src.aws.opensearch_client import get_opensearch_client, get_index_name
from src.aws.bedrock_client import get_embeddings
from src.retrieval.state import RetrievedChunk

load_dotenv()


class StratifiedHybridRetriever:
    """
    Performs stratified hybrid retrieval per source type.
    
    Strategy:
    1. For each source_type (filing, call, chat):
       - Run BM25 keyword search (top kb)
       - Run kNN vector search (top kv)
       - Merge via Reciprocal Rank Fusion
       - Keep top k_source results
    2. Merge all sources with source-specific weights
    3. Apply recency boost and diversity constraints
    """
    
    def __init__(
        self,
        k_per_source: int = 5,
        vector_weight: float = 0.6,
        keyword_weight: float = 0.4,
        source_weights: Optional[Dict[str, float]] = None,
        recency_halflife_days: int = 14
    ):
        """
        Initialize retriever.
        
        Args:
            k_per_source: Top-k results to keep per source type
            vector_weight: Weight for vector search (0-1)
            keyword_weight: Weight for keyword search (0-1)
            source_weights: Weights for each source type
            recency_halflife_days: Halflife for recency decay
        """
        self.k_per_source = k_per_source
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        
        self.source_weights = source_weights or {
            "filing": 0.35,
            "call": 0.30,
            "chat": 0.25,
        }
        
        self.recency_halflife_days = recency_halflife_days
        
        # Initialize clients
        self.os_client = get_opensearch_client()
        self.index_name = get_index_name()
        self.embeddings = get_embeddings()
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to 0-1 range using min-max normalization."""
        if not scores or len(scores) == 0:
            return []
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [1.0] * len(scores)
        
        return [(s - min_score) / (max_score - min_score) for s in scores]
    
    def _reciprocal_rank_fusion(
        self,
        results_list: List[List[Tuple[dict, float]]],
        k: int = 60
    ) -> List[Tuple[dict, float]]:
        """
        Combine multiple ranked lists using Reciprocal Rank Fusion.
        
        Args:
            results_list: List of ranked result lists [(doc, score), ...]
            k: Constant for RRF formula (default 60)
            
        Returns:
            Combined ranked list
        """
        doc_scores = {}
        
        for results in results_list:
            for rank, (doc, score) in enumerate(results, start=1):
                # Use chunk_id from metadata or id as unique identifier
                metadata = doc.get('metadata', {})
                doc_id = metadata.get('chunk_id') or doc.get('id') or str(doc)[:100]
                
                rrf_score = 1.0 / (k + rank)
                
                if doc_id in doc_scores:
                    doc_scores[doc_id]['score'] += rrf_score
                else:
                    doc_scores[doc_id] = {
                        'doc': doc,
                        'score': rrf_score,
                        'original_score': score
                    }
        
        # Sort by RRF score
        sorted_docs = sorted(
            doc_scores.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )
        
        return [(item[1]['doc'], item[1]['score']) for item in sorted_docs]
    
    def _keyword_search(
        self,
        query: str,
        source_type: str,
        k: int,
        filters: Optional[dict] = None
    ) -> List[Tuple[dict, float]]:
        """Perform BM25 keyword search."""
        # Build query
        must_clauses = [
            {"match": {"text": {"query": query, "fuzziness": "AUTO"}}}
        ]

        # Add source type filter - metadata is nested, use .keyword for text fields
        filter_clauses = [
            {"term": {"metadata.source_type.keyword": source_type}}
        ]

        # Add additional filters with metadata prefix and .keyword for text fields
        if filters:
            for field, value in filters.items():
                filter_clauses.append({"term": {f"metadata.{field}.keyword": value}})
        
        search_body = {
            "query": {
                "bool": {
                    "must": must_clauses,
                    "filter": filter_clauses
                }
            },
            "size": k * 2  # Fetch more for RRF
        }
        
        try:
            response = self.os_client.search(
                index=self.index_name,
                body=search_body
            )
            
            hits = response['hits']['hits']
            return [(hit['_source'], hit['_score']) for hit in hits]
        
        except Exception as e:
            print(f"Keyword search error: {e}")
            return []
    
    def _vector_search(
        self,
        query: str,
        source_type: str,
        k: int,
        filters: Optional[dict] = None
    ) -> List[Tuple[dict, float]]:
        """Perform kNN vector search."""
        # Generate query embedding
        query_embedding = self.embeddings.embed_query(query)

        # Build filter with metadata prefix and .keyword for text fields
        filter_clauses = [
            {"term": {"metadata.source_type.keyword": source_type}}
        ]

        if filters:
            for field, value in filters.items():
                filter_clauses.append({"term": {f"metadata.{field}.keyword": value}})
        
        search_body = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "knn": {
                                "vector_field": {  # Changed from embedding to vector_field
                                    "vector": query_embedding,
                                    "k": k * 2  # Fetch more for RRF
                                }
                            }
                        }
                    ],
                    "filter": filter_clauses
                }
            },
            "size": k * 2
        }
        
        try:
            response = self.os_client.search(
                index=self.index_name,
                body=search_body
            )
            
            hits = response['hits']['hits']
            return [(hit['_source'], hit['_score']) for hit in hits]
        
        except Exception as e:
            print(f"Vector search error: {e}")
            return []
    
    def _calculate_recency_score(self, doc_metadata: dict) -> float:
        """Calculate recency score with exponential decay based on source type."""
        try:
            # In the current schema, all dates are stored in 'timestamp' field
            date_str = doc_metadata.get('timestamp')

            if not date_str:
                return 1.0

            doc_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            now = datetime.now(doc_date.tzinfo)

            age_days = (now - doc_date).days

            # Exponential decay: score = exp(-age / halflife)
            decay_factor = math.exp(-age_days / self.recency_halflife_days)

            return max(0.1, decay_factor)  # Minimum score of 0.1

        except Exception:
            return 1.0  # Default score if date parsing fails
    
    def _search_source_type(
        self,
        query: str,
        source_type: str,
        filters: Optional[dict] = None
    ) -> List[RetrievedChunk]:
        """
        Hybrid search for a specific source type.
        
        Returns:
            Top-k retrieved chunks for this source
        """
        # Keyword search
        keyword_results = self._keyword_search(
            query, source_type, self.k_per_source, filters
        )
        
        # Vector search
        vector_results = self._vector_search(
            query, source_type, self.k_per_source, filters
        )
        
        # Reciprocal Rank Fusion
        fused_results = self._reciprocal_rank_fusion([
            keyword_results,
            vector_results
        ])
        
        # Convert to RetrievedChunk objects
        chunks = []
        for doc, rrf_score in fused_results[:self.k_per_source]:
            # Extract metadata (it's nested in the document)
            metadata = doc.get('metadata', {})

            # Get chunk_id from metadata for matching
            chunk_id = metadata.get('chunk_id') or doc.get('id', '')

            # Calculate normalized scores
            keyword_score = next(
                (s for d, s in keyword_results
                 if (d.get('metadata', {}).get('chunk_id') or d.get('id')) == chunk_id),
                0.0
            )
            vector_score = next(
                (s for d, s in vector_results
                 if (d.get('metadata', {}).get('chunk_id') or d.get('id')) == chunk_id),
                0.0
            )
            
            # Hybrid score
            hybrid_score = (
                self.vector_weight * vector_score +
                self.keyword_weight * keyword_score
            )
            
            # Recency score - use metadata for date fields
            recency_score = self._calculate_recency_score(metadata)

            # Final score
            final_score = hybrid_score * recency_score

            # Get timestamp based on source type
            timestamp_value = metadata.get('timestamp')
            filing_date = None
            call_date = None
            chat_timestamp = None

            if source_type == 'filing' and timestamp_value:
                filing_date = timestamp_value
            elif source_type == 'call' and timestamp_value:
                call_date = timestamp_value
            elif source_type == 'chat' and timestamp_value:
                chat_timestamp = timestamp_value

            chunk = RetrievedChunk(
                chunk_id=metadata.get('chunk_id', chunk_id),
                parent_doc_id=metadata.get('parent_doc_id', ''),
                text=doc.get('text', ''),
                source_type=source_type,
                ticker=metadata.get('ticker', ''),
                company_name=metadata.get('company_name', ''),
                # Temporal fields
                fiscal_year=metadata.get('fiscal_year'),
                fiscal_quarter=metadata.get('fiscal_quarter'),
                fiscal_period=metadata.get('fiscal_period'),
                # Date fields
                filing_date=filing_date,
                call_date=call_date,
                timestamp=chat_timestamp,
                # Scores
                vector_score=vector_score,
                keyword_score=keyword_score,
                hybrid_score=hybrid_score,
                recency_score=recency_score,
                final_score=final_score,
                # Filing-specific
                filing_type=metadata.get('filing_type'),
                section=metadata.get('section'),
                has_tables=metadata.get('has_tables'),
                # Call-specific
                speaker=metadata.get('speaker'),
                speaker_role=metadata.get('speaker_role'),
                is_company_speaker=metadata.get('is_company_speaker'),
                # Chat-specific
                trader_id=metadata.get('trader_id'),
                trader_role=metadata.get('trader_role'),
                message_index=metadata.get('message_index'),
                # Enrichment fields
                contains_numbers=metadata.get('contains_numbers'),
                forward_looking=metadata.get('forward_looking'),
                importance_score=metadata.get('importance_score'),
                # LLM-enhanced fields
                sentiment=metadata.get('sentiment'),
                sentiment_score=metadata.get('sentiment_score'),
                confidence_level=metadata.get('confidence_level'),
                # Chat signals
                signal_type=metadata.get('signal_type'),
                credibility_score=metadata.get('credibility_score'),
                actionability=metadata.get('actionability'),
                urgency=metadata.get('urgency'),
            )
            
            chunks.append(chunk)
        
        return chunks
    
    def retrieve(
        self,
        query: str,
        ticker: Optional[str] = None,
        fiscal_period: Optional[str] = None,
        source_types: Optional[List[str]] = None
    ) -> List[RetrievedChunk]:
        """
        Perform stratified hybrid retrieval across all sources.
        
        Args:
            query: Search query
            ticker: Optional ticker filter
            fiscal_period: Optional fiscal period filter
            source_types: Optional list of source types to search
            
        Returns:
            Ranked list of retrieved chunks
        """
        # Build filters
        filters = {}
        if ticker:
            filters['ticker'] = ticker
        if fiscal_period:
            filters['fiscal_period'] = fiscal_period
        
        # Determine source types to search
        if source_types is None:
            source_types = ["filing", "call", "chat"]
        
        # Retrieve from each source
        all_chunks = []
        for source_type in source_types:
            chunks = self._search_source_type(query, source_type, filters)
            
            # Apply source weight
            source_weight = self.source_weights.get(source_type, 0.33)
            for chunk in chunks:
                chunk.final_score *= source_weight
            
            all_chunks.extend(chunks)
        
        # Sort by final score
        all_chunks.sort(key=lambda x: x.final_score, reverse=True)
        
        return all_chunks