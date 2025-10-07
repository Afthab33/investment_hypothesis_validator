#!/usr/bin/env python3
"""
Fixed version with explicit index creation and better error handling
"""

import os
import json
import boto3
import time
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_aws import BedrockEmbeddings
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth

load_dotenv()


def setup_aws_auth(region: str):
    """Setup AWS authentication"""
    credentials = boto3.Session(region_name=region).get_credentials()
    return AWS4Auth(
        credentials.access_key,
        credentials.secret_key,
        region,
        "aoss",
        session_token=credentials.token
    )


def create_opensearch_client(url: str, auth, region: str):
    """Create OpenSearch client"""
    # Extract host from URL
    host = url.replace('https://', '').replace('http://', '')

    client = OpenSearch(
        hosts=[{'host': host, 'port': 443}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        timeout=300
    )
    return client


def check_index_exists(client, index_name: str) -> bool:
    """Check if index exists"""
    try:
        return client.indices.exists(index=index_name)
    except Exception as e:
        print(f"Error checking index: {e}")
        return False


def delete_index_if_exists(client, index_name: str):
    """Delete index if it exists"""
    if check_index_exists(client, index_name):
        print(f"⚠️  Index '{index_name}' already exists. Deleting...")
        try:
            client.indices.delete(index=index_name)
            print(f"✅ Deleted existing index")
            time.sleep(2)  # Wait for deletion to complete
        except Exception as e:
            print(f"❌ Failed to delete index: {e}")


def load_all_data() -> List[Dict]:
    """Load all finalized datasets"""
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)

    files = {
        "filings": "data/tesla/tesla_filings/all_filings_finalized.json",
        "calls": "data/tesla/tesla_earning_calls/processed/calls_finalized.json",
        "chats": "data/tesla/bloomberg_chats_synthetic/processed/bloomberg_chunks_with_signals.json"
    }

    all_chunks = []
    for name, path in files.items():
        with open(path) as f:
            chunks = json.load(f)
        print(f"✅ {name}: {len(chunks)} chunks")
        all_chunks.extend(chunks)

    print(f"\n📊 Total: {len(all_chunks)} chunks")
    return all_chunks


def prepare_documents(chunks: List[Dict]) -> List[Document]:
    """Convert to LangChain Documents"""
    print("\n" + "="*80)
    print("PREPARING DOCUMENTS")
    print("="*80)

    documents = []
    for chunk in chunks:
        text = chunk.get('text', '')
        if not text:
            continue

        metadata = {
            "chunk_id": chunk.get('chunk_id'),
            "parent_doc_id": chunk.get('parent_doc_id'),
            "source_type": chunk.get('source_type'),
            "ticker": chunk.get('ticker'),
            "company_name": chunk.get('company_name'),
            "fiscal_year": chunk.get('fiscal_year'),
            "fiscal_quarter": chunk.get('fiscal_quarter'),
            "fiscal_period": chunk.get('fiscal_period'),
            "section": chunk.get('section'),
            "speaker": chunk.get('speaker'),
            "speaker_role": chunk.get('speaker_role'),
            "trader_role": chunk.get('trader_role'),
            "importance_score": chunk.get('importance_score'),
            "credibility_score": chunk.get('credibility_score'),
            "contains_numbers": chunk.get('contains_numbers'),
            "forward_looking": chunk.get('forward_looking'),
            "sentiment": chunk.get('tone_analysis', {}).get('sentiment') if chunk.get('tone_analysis') else None,
            "sentiment_score": chunk.get('tone_analysis', {}).get('sentiment_score') if chunk.get('tone_analysis') else None,
            "signal_type": chunk.get('signal_type'),
            "actionability": chunk.get('actionability'),
            "urgency": chunk.get('urgency'),
            "timestamp": chunk.get('timestamp') or chunk.get('call_date') or chunk.get('filing_date'),
        }

        # Remove None values
        metadata = {k: v for k, v in metadata.items() if v is not None}

        documents.append(Document(page_content=text, metadata=metadata))

    print(f"✅ Prepared {len(documents)} documents")
    return documents


def ingest_with_retry(documents: List[Document], embeddings, client, index_name: str, awsauth, opensearch_url: str):
    """Ingest with small batches and retry logic"""
    print("\n" + "="*80)
    print("INGESTING TO OPENSEARCH")
    print("="*80)

    batch_size = 50  # Small batches to avoid timeouts
    total = len(documents)

    print(f"\n📤 Processing {total} documents in batches of {batch_size}...")

    # Process first batch to create index
    print(f"\n[1/{(total-1)//batch_size + 1}] Processing first batch (creates index)...")
    first_batch = documents[:batch_size]

    try:
        vectorstore = OpenSearchVectorSearch.from_documents(
            first_batch,
            embeddings,
            opensearch_url=opensearch_url,
            http_auth=awsauth,
            timeout=300,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            index_name=index_name,
            engine="faiss",
            bulk_size=batch_size,
        )
        print(f"✅ First batch ingested, index created")
        time.sleep(2)

        # Verify index was created
        if not check_index_exists(client, index_name):
            raise Exception(f"Index {index_name} was not created!")

        print(f"✅ Verified index exists")

    except Exception as e:
        print(f"❌ Failed to create index: {e}")
        raise

    # Process remaining batches
    remaining = documents[batch_size:]
    batch_num = 2

    for i in range(0, len(remaining), batch_size):
        batch = remaining[i:i+batch_size]
        print(f"\n[{batch_num}/{(total-1)//batch_size + 1}] Adding {len(batch)} documents...")

        try:
            vectorstore.add_documents(batch)
            print(f"✅ Batch {batch_num} added")
            batch_num += 1
            time.sleep(1)  # Small delay between batches

        except Exception as e:
            print(f"❌ Batch {batch_num} failed: {e}")
            print("Retrying...")
            try:
                time.sleep(3)
                vectorstore.add_documents(batch)
                print(f"✅ Retry successful")
                batch_num += 1
            except Exception as e2:
                print(f"❌ Retry failed: {e2}")
                raise

    return vectorstore


def verify_ingestion(vectorstore: OpenSearchVectorSearch):
    """Test searches"""
    print("\n" + "="*80)
    print("VERIFICATION")
    print("="*80)

    queries = [
        "What are Tesla's risk factors?",
        "What did the CEO say about profitability?",
        "Bloomberg chat about margins"
    ]

    for i, query in enumerate(queries, 1):
        print(f"\n{i}. '{query}'")
        try:
            results = vectorstore.similarity_search(query, k=3)
            for j, doc in enumerate(results, 1):
                print(f"   [{j}] {doc.metadata.get('source_type')} - {doc.metadata.get('chunk_id')}")
                print(f"       {doc.page_content[:80]}...")
        except Exception as e:
            print(f"   ❌ Failed: {e}")


def main():
    print("\n" + "="*80)
    print("OPENSEARCH INGESTION - FIXED VERSION")
    print("="*80)

    # Config
    OPENSEARCH_URL = os.getenv("OPENSEARCH_URL")
    INDEX_NAME = os.getenv("OPENSEARCH_INDEX_NAME", "investment-documents")
    AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

    if not OPENSEARCH_URL:
        print("❌ OPENSEARCH_URL not in .env")
        return

    print(f"\n📋 Config:")
    print(f"   URL: {OPENSEARCH_URL}")
    print(f"   Index: {INDEX_NAME}")
    print(f"   Region: {AWS_REGION}")

    # Setup
    awsauth = setup_aws_auth(AWS_REGION)
    client = create_opensearch_client(OPENSEARCH_URL, awsauth, AWS_REGION)
    embeddings = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v2:0",
        region_name=AWS_REGION
    )

    # Check/delete existing index
    delete_index_if_exists(client, INDEX_NAME)

    # Load data
    chunks = load_all_data()
    documents = prepare_documents(chunks)

    # Ingest
    vectorstore = ingest_with_retry(
        documents, embeddings, client, INDEX_NAME, awsauth, OPENSEARCH_URL
    )

    # Verify
    verify_ingestion(vectorstore)

    # Stats
    print("\n" + "="*80)
    print("FINAL STATS")
    print("="*80)

    stats = client.indices.stats(index=INDEX_NAME)
    doc_count = stats['indices'][INDEX_NAME]['total']['docs']['count']
    print(f"\n✅ Index: {INDEX_NAME}")
    print(f"✅ Documents in index: {doc_count}")
    print(f"✅ Expected: {len(documents)}")

    if doc_count == len(documents):
        print(f"\n🎉 SUCCESS! All {doc_count} documents ingested!")
    else:
        print(f"\n⚠️  Mismatch: {doc_count} in index vs {len(documents)} expected")


if __name__ == "__main__":
    main()
