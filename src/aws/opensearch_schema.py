"""
OpenSearch Index Schema for Soros Investment Validator
Demonstrates understanding of OpenSearch optimization and investment data structure
"""

SOROS_DOCUMENT_INDEX = {
    "settings": {
        "index": {
            # Vector search settings
            "knn": True,
            "knn.algo_param.ef_search": 256,

            # Performance optimization
            "number_of_shards": 3,
            "number_of_replicas": 1,
            "refresh_interval": "30s",

            # Relevance tuning
            "max_result_window": 10000,
            "max_inner_result_window": 5000
        },
        "analysis": {
            "analyzer": {
                "soros_text_analyzer": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "char_filter": ["html_strip"],
                    "filter": [
                        "lowercase",
                        "asciifolding",
                        "financial_synonyms",
                        "ticker_normalizer",
                        "english_stop",
                        "english_stemmer"
                    ]
                },
                "exact_match_analyzer": {
                    "type": "keyword",
                    "filter": ["lowercase"]
                }
            },
            "filter": {
                "financial_synonyms": {
                    "type": "synonym_graph",
                    "synonyms": [
                        # Financial metrics
                        "eps, earnings per share => earnings_per_share",
                        "p/e, pe, price earnings => price_earnings_ratio",
                        "fcf, free cash flow => free_cash_flow",
                        "capex, capital expenditure => capital_expenditure",
                        "opex, operating expense => operating_expense",
                        "cogs, cost of goods sold => cost_of_goods_sold",
                        "sga, sg&a => selling_general_administrative",

                        # Business terms
                        "m&a, merger, acquisition => merger_acquisition",
                        "r&d, research, development => research_development",
                        "ai, artificial intelligence => artificial_intelligence",
                        "ev, electric vehicle => electric_vehicle",
                        "fsd, full self driving => full_self_driving",

                        # Sentiment terms
                        "headwind, challenge, pressure => negative_pressure",
                        "tailwind, opportunity, catalyst => positive_catalyst",
                        "guidance, outlook, forecast => forward_guidance",
                        "beat, exceed, outperform => positive_surprise",
                        "miss, disappoint, underperform => negative_surprise"
                    ]
                },
                "ticker_normalizer": {
                    "type": "pattern_replace",
                    "pattern": "\\$([A-Z]+)",
                    "replacement": "$1 ticker_$1"
                },
                "english_stop": {
                    "type": "stop",
                    "stopwords": "_english_"
                },
                "english_stemmer": {
                    "type": "stemmer",
                    "language": "english"
                }
            }
        }
    },
    "mappings": {
        "properties": {
            # ============= Core Document Fields =============
            "doc_id": {
                "type": "keyword",
                "doc_values": True
            },
            "chunk_id": {
                "type": "keyword",
                "doc_values": True
            },
            "parent_doc_id": {
                "type": "keyword",
                "doc_values": True
            },

            # ============= Classification Fields =============
            "source_type": {
                "type": "keyword",  # "filing", "call", "chat", "8k"
                "doc_values": True
            },
            "document_type": {
                "type": "keyword",  # "10-Q", "10-K", "8-K", "earnings_call", "ib_chat"
                "doc_values": True
            },
            "ticker": {
                "type": "keyword",
                "doc_values": True,
                "fields": {
                    "search": {
                        "type": "text",
                        "analyzer": "exact_match_analyzer"
                    }
                }
            },
            "company_name": {
                "type": "keyword",
                "fields": {
                    "text": {
                        "type": "text",
                        "analyzer": "standard"
                    }
                }
            },

            # ============= Temporal Fields =============
            "fiscal_year": {
                "type": "integer",
                "doc_values": True
            },
            "fiscal_quarter": {
                "type": "keyword",  # "Q1", "Q2", "Q3", "Q4", "FY"
                "doc_values": True
            },
            "fiscal_period": {
                "type": "keyword",  # "2024Q2", "FY2024"
                "doc_values": True
            },
            "document_date": {
                "type": "date",
                "format": "yyyy-MM-dd||yyyy-MM-dd'T'HH:mm:ss.SSS'Z'",
                "doc_values": True
            },
            "ingestion_timestamp": {
                "type": "date",
                "doc_values": True
            },

            # ============= Content Fields =============
            "text": {
                "type": "text",
                "analyzer": "soros_text_analyzer",
                "term_vector": "with_positions_offsets",
                "fields": {
                    "exact": {
                        "type": "text",
                        "analyzer": "exact_match_analyzer"
                    },
                    "keyword": {
                        "type": "keyword",
                        "ignore_above": 256
                    }
                }
            },
            "text_length": {
                "type": "integer",
                "doc_values": True
            },
            "embedding": {
                "type": "knn_vector",
                "dimension": 1536,  # Bedrock Titan embedding size
                "method": {
                    "name": "hnsw",
                    "space_type": "cosinesimil",
                    "engine": "nmslib",
                    "parameters": {
                        "ef_construction": 256,
                        "m": 32
                    }
                }
            },

            # ============= Chunk Context Fields =============
            "chunk_metadata": {
                "type": "object",
                "properties": {
                    "position": {"type": "integer"},
                    "total_chunks": {"type": "integer"},
                    "chunk_size": {"type": "integer"},
                    "overlap_size": {"type": "integer"},
                    "prev_chunk_id": {"type": "keyword"},
                    "next_chunk_id": {"type": "keyword"},
                    "is_continuation": {"type": "boolean"},
                    "requires_context": {"type": "boolean"}
                }
            },

            # ============= Filing-Specific Fields =============
            "filing_metadata": {
                "type": "object",
                "properties": {
                    "section": {"type": "keyword"},  # "MD&A", "Risk Factors", etc.
                    "subsection": {"type": "keyword"},
                    "page_number": {"type": "integer"},
                    "total_pages": {"type": "integer"},
                    "is_financial_table": {"type": "boolean"},
                    "contains_forward_looking": {"type": "boolean"},
                    "contains_non_gaap": {"type": "boolean"}
                }
            },

            # ============= Earnings Call-Specific Fields =============
            "call_metadata": {
                "type": "object",
                "properties": {
                    "speaker_name": {"type": "keyword"},
                    "speaker_role": {"type": "keyword"},  # "CEO", "CFO", "Analyst"
                    "speaker_company": {"type": "keyword"},
                    "call_section": {"type": "keyword"},  # "prepared_remarks", "Q&A"
                    "turn_number": {"type": "integer"},
                    "question_id": {"type": "keyword"},  # Links Q&A pairs
                    "is_company_speaker": {"type": "boolean"}
                }
            },

            # ============= Tonal Analysis Fields (for Soros JD) =============
            "tone_analysis": {
                "type": "object",
                "properties": {
                    "sentiment": {"type": "keyword"},  # "bullish", "neutral", "bearish"
                    "sentiment_score": {"type": "float"},  # -1.0 to 1.0
                    "confidence_level": {"type": "float"},  # 0.0 to 1.0
                    "uncertainty_score": {"type": "float"},
                    "forward_looking_score": {"type": "float"},
                    "key_phrases": {"type": "keyword"},  # Important extracted phrases
                    "hedge_words_count": {"type": "integer"},
                    "certainty_words_count": {"type": "integer"}
                }
            },

            # ============= Bloomberg Chat-Specific Fields =============
            "chat_metadata": {
                "type": "object",
                "properties": {
                    "trader_id": {"type": "keyword"},
                    "message_timestamp": {"type": "date"},
                    "chat_room": {"type": "keyword"},
                    "mentioned_tickers": {"type": "keyword"},  # Array of tickers
                    "message_type": {"type": "keyword"},  # "rumor", "analysis", "news"
                    "credibility_score": {"type": "float"},
                    "actionability": {"type": "keyword"}  # "high", "medium", "low"
                }
            },

            # ============= Financial Metrics Extraction =============
            "extracted_metrics": {
                "type": "nested",
                "properties": {
                    "metric_name": {"type": "keyword"},  # "gross_margin", "revenue"
                    "metric_value": {"type": "float"},
                    "metric_unit": {"type": "keyword"},  # "percent", "millions", "billions"
                    "metric_period": {"type": "keyword"},  # "Q2", "YTD", "FY"
                    "is_guidance": {"type": "boolean"},
                    "comparison_period": {"type": "keyword"},
                    "comparison_value": {"type": "float"},
                    "yoy_change": {"type": "float"}
                }
            },

            # ============= Search & Retrieval Optimization =============
            "importance_score": {
                "type": "float",  # 0.0 to 1.0, for ranking
                "doc_values": True
            },
            "recency_score": {
                "type": "float",  # Decay function based on age
                "doc_values": True
            },
            "relevance_keywords": {
                "type": "keyword",  # Top TF-IDF terms
                "doc_values": True
            },
            "has_material_info": {
                "type": "boolean",
                "doc_values": True
            },

            # ============= Audit & Compliance =============
            "processing_metadata": {
                "type": "object",
                "enabled": False,  # Not searchable, just stored
                "properties": {
                    "pipeline_version": {"type": "keyword"},
                    "model_version": {"type": "keyword"},
                    "processing_time_ms": {"type": "long"},
                    "chunk_method": {"type": "keyword"},
                    "source_checksum": {"type": "keyword"}
                }
            }
        }
    }
}