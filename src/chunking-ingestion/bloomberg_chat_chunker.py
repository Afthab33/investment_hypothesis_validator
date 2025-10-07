#!/usr/bin/env python3
"""
Bloomberg Chat Chunker - Option 1: Individual Message Chunking
Each chat message becomes a separate chunk with LLM signal extraction
"""

import os
import json
import re
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dotenv import load_dotenv

from openai_llm_analyzer import OpenAISignalExtractor

load_dotenv()


class BloombergChatChunker:
    """
    Chunks Bloomberg chat files into individual messages
    Each message = 1 chunk for optimal RAG retrieval
    """

    def __init__(self, use_llm: bool = True):
        """
        Args:
            use_llm: If True, applies LLM signal extraction during chunking
        """
        self.use_llm = use_llm
        self.signal_extractor = OpenAISignalExtractor() if use_llm else None

        # Trader role inference rules
        self.role_patterns = {
            "PM": "portfolio_manager",
            "Analyst": "analyst",
            "Trader": "trader",
            "Senior": "senior_trader",
            "Junior": "junior_trader"
        }

    def parse_chat_file(self, file_path: str) -> Dict:
        """
        Parse a Bloomberg chat file into metadata + messages

        Returns:
            {
                'metadata': {...},
                'messages': [...]
            }
        """
        with open(file_path, 'r') as f:
            content = f.read()

        # Split header (JSON) from messages
        parts = content.split('---', 1)

        if len(parts) != 2:
            raise ValueError(f"Invalid chat file format: {file_path}")

        # Parse metadata
        metadata = json.loads(parts[0].strip())

        # Parse messages
        message_lines = parts[1].strip().split('\n')
        messages = []

        for line in message_lines:
            line = line.strip()
            if not line:
                continue

            # Expected format: "Trader1: message text"
            if ':' in line:
                trader_id, text = line.split(':', 1)
                messages.append({
                    'trader_id': trader_id.strip(),
                    'text': text.strip()
                })

        return {
            'metadata': metadata,
            'messages': messages
        }

    def infer_trader_role(self, trader_id: str) -> str:
        """
        Infer trader role from trader_id

        Examples:
            PM -> portfolio_manager
            Analyst1 -> analyst
            Trader2 -> trader
        """
        for pattern, role in self.role_patterns.items():
            if pattern in trader_id:
                return role

        # Default to trader
        return "trader"

    def estimate_timestamp(
        self,
        base_date: str,
        message_index: int,
        total_messages: int
    ) -> str:
        """
        Estimate timestamp for a message within a chat session

        Args:
            base_date: Starting date from metadata (e.g., "2024-07-20")
            message_index: Position of message in conversation
            total_messages: Total number of messages

        Returns:
            ISO format timestamp
        """
        # Parse base date (handle ranges like "2024-07-20 to 2024-07-23")
        if " to " in base_date:
            base_date = base_date.split(" to ")[0]

        # Parse date
        date_obj = datetime.strptime(base_date, "%Y-%m-%d")

        # Assume chat session spans 4 hours (9:30 AM - 1:30 PM)
        # Distribute messages evenly across this window
        session_start = date_obj.replace(hour=9, minute=30, second=0)

        if total_messages > 1:
            minutes_offset = int((message_index / (total_messages - 1)) * 240)  # 240 min = 4 hrs
        else:
            minutes_offset = 0

        timestamp = session_start + timedelta(minutes=minutes_offset)

        return timestamp.isoformat() + "Z"

    def create_chunk(
        self,
        message: Dict,
        metadata: Dict,
        message_index: int,
        total_messages: int
    ) -> Dict:
        """
        Create a chunk from a single message

        Args:
            message: {'trader_id': '...', 'text': '...'}
            metadata: Chat file metadata
            message_index: Position in conversation
            total_messages: Total messages in chat

        Returns:
            Chunk dict with all required fields
        """
        doc_id = metadata.get('doc_id', 'unknown')

        chunk = {
            # Identifiers
            "chunk_id": f"{doc_id}_msg_{message_index:03d}",
            "parent_doc_id": doc_id,

            # Core content
            "text": message['text'],
            "trader_id": message['trader_id'],
            "trader_role": self.infer_trader_role(message['trader_id']),

            # Metadata
            "source_type": "chat",
            "source_name": metadata.get('source_name', 'Bloomberg IB'),
            "ticker": metadata.get('tickers', ['UNKNOWN'])[0] if metadata.get('tickers') else 'UNKNOWN',
            "company_name": metadata.get('company', 'Unknown'),
            "timestamp": self.estimate_timestamp(
                metadata.get('date_range', '2024-01-01'),
                message_index,
                total_messages
            ),

            # Chat-specific fields
            "message_index": message_index,
            "total_messages_in_chat": total_messages,
            "language": metadata.get('language', 'en'),
            "region": metadata.get('region', 'US'),

            # Placeholders for LLM extraction (if not using LLM now)
            "signal_type": None,
            "credibility_score": None,
            "actionability": None,
            "urgency": None,
            "has_price_target": None,
            "price_target": None,
            "mentioned_tickers": None,
            "confidence_in_extraction": None,
            "reasoning": None
        }

        return chunk

    def chunk_chat_file(self, file_path: str) -> List[Dict]:
        """
        Process a single chat file into chunks
        Applies LLM signal extraction if enabled

        Returns:
            List of chunk dicts
        """
        print(f"\nProcessing: {file_path}")

        # Parse file
        parsed = self.parse_chat_file(file_path)
        metadata = parsed['metadata']
        messages = parsed['messages']

        print(f"  Found {len(messages)} messages")

        # Create chunks
        chunks = []
        total_messages = len(messages)

        for i, message in enumerate(messages):
            # Create base chunk
            chunk = self.create_chunk(message, metadata, i, total_messages)

            # Apply LLM signal extraction if enabled
            if self.use_llm and self.signal_extractor:
                print(f"    Extracting signals from message {i+1}/{total_messages}...", end=" ")

                try:
                    signal = self.signal_extractor.extract_signal(
                        message=message['text'],
                        trader_id=chunk['trader_id'],
                        trader_role=chunk['trader_role']
                    )

                    # Add LLM results to chunk
                    chunk['signal_type'] = signal.signal_type
                    chunk['credibility_score'] = signal.credibility_score
                    chunk['actionability'] = signal.actionability
                    chunk['urgency'] = signal.urgency
                    chunk['has_price_target'] = signal.has_price_target
                    chunk['price_target'] = signal.price_target
                    chunk['mentioned_tickers'] = signal.mentioned_tickers
                    chunk['confidence_in_extraction'] = signal.confidence_in_extraction
                    chunk['reasoning'] = signal.reasoning

                    print(f"✓ [{signal.signal_type}]")

                except Exception as e:
                    print(f"✗ Error: {e}")

            chunks.append(chunk)

        return chunks

    def process_all_chats(self, chat_dir: str, output_file: str) -> List[Dict]:
        """
        Process all chat files in a directory

        Args:
            chat_dir: Directory containing chat files
            output_file: Where to save processed chunks

        Returns:
            List of all chunks from all chat files
        """
        chat_files = list(Path(chat_dir).glob("*.txt"))

        if not chat_files:
            print(f"No chat files found in {chat_dir}")
            return []

        print(f"\n{'='*60}")
        print(f"BLOOMBERG CHAT CHUNKING - Option 1: Individual Messages")
        print(f"{'='*60}")
        print(f"Found {len(chat_files)} chat files")
        print(f"LLM Signal Extraction: {'ENABLED' if self.use_llm else 'DISABLED'}")

        all_chunks = []

        for chat_file in sorted(chat_files):
            chunks = self.chunk_chat_file(str(chat_file))
            all_chunks.extend(chunks)

        # Save results
        print(f"\n{'='*60}")
        print(f"PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"Total chunks created: {len(all_chunks)}")

        # Save to file
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(all_chunks, f, indent=2)

        print(f"Saved to: {output_file}")

        # Show sample results
        if all_chunks and self.use_llm:
            print(f"\n{'='*60}")
            print("SAMPLE SIGNAL EXTRACTION RESULTS")
            print(f"{'='*60}")

            for i, chunk in enumerate(all_chunks[:5]):
                print(f"\nMessage {i+1}:")
                print(f"  Trader: {chunk['trader_id']} ({chunk['trader_role']})")
                print(f"  Text: {chunk['text']}")
                print(f"  Signal Type: {chunk.get('signal_type', 'N/A')}")
                cred = chunk.get('credibility_score')
                print(f"  Credibility: {cred:.2f}" if cred is not None else "  Credibility: N/A")
                print(f"  Actionability: {chunk.get('actionability', 'N/A')}")
                print(f"  Urgency: {chunk.get('urgency', 'N/A')}")
                print(f"  Price Target: {chunk.get('price_target', 'N/A')}")
                print(f"  Reasoning: {chunk.get('reasoning', 'N/A')}")

        return all_chunks


def main():
    """Main execution"""

    # Configuration
    chat_dir = "data/tesla/bloomberg_chats_synthetic"
    output_file = "data/tesla/bloomberg_chats_synthetic/processed/bloomberg_chunks_with_signals.json"

    # Check OpenAI key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found in .env file")
        print("Set use_llm=False to chunk without LLM extraction")
        return

    print("✅ OpenAI API key found")

    # Initialize chunker with LLM enabled
    chunker = BloombergChatChunker(use_llm=True)

    # Process all chats
    chunks = chunker.process_all_chats(chat_dir, output_file)

    print(f"\n✅ Successfully processed {len(chunks)} chat messages")
    print(f"\n💡 Next steps:")
    print("1. Review the generated chunks")
    print("2. Validate signal extraction accuracy")
    print("3. Transform to finalized schema if needed")
    print("4. Ingest into OpenSearch")


if __name__ == "__main__":
    main()
