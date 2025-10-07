#!/usr/bin/env python3
"""
Earnings Call Processor with LLM Tonal Analysis
Transforms raw chunks to finalized schema with OpenAI-based sentiment analysis
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from dotenv import load_dotenv

from openai_llm_analyzer import OpenAITonalAnalyzer

load_dotenv()


class EarningsCallProcessor:
    """
    Process earnings call chunks with LLM tonal analysis
    Transform to finalized schema aligned with FINALIZED_CALL_SCHEMA.md
    """

    def __init__(self, use_llm: bool = True):
        """
        Args:
            use_llm: If True, applies OpenAI tonal analysis
        """
        self.use_llm = use_llm
        self.tonal_analyzer = OpenAITonalAnalyzer() if use_llm else None

        # Speaker role mapping
        self.speaker_role_map = {
            "CEO": ["chief executive", "ceo", "elon musk"],
            "CFO": ["chief financial", "cfo", "vaibhav taneja"],
            "Executive": ["vice president", "director", "vp", "head", "lars moravy", "ashok elluswamy"],
            "Analyst": ["analyst", "morgan stanley", "goldman sachs", "bernstein", "piper sandler",
                       "oppenheimer", "baird", "wolfe research"],
            "Operator": ["operator", "conference"],
            "IR": ["investor relations", "ir", "martin viecha"]
        }

    def normalize_speaker_role(self, speaker: str, current_role: str = None) -> tuple:
        """
        Normalize speaker role to standard categories
        Returns: (normalized_role, is_company_speaker)
        """
        speaker_lower = speaker.lower()

        # Check role mappings
        for role, keywords in self.speaker_role_map.items():
            if any(keyword in speaker_lower for keyword in keywords):
                is_company = role in ["CEO", "CFO", "Executive", "IR"]
                return role, is_company

        # Default
        return current_role or "Unknown", False

    def normalize_section(self, section: str) -> str:
        """Normalize section names to standard values"""
        section_lower = section.lower()

        if any(word in section_lower for word in ["opening", "introduction", "participants"]):
            return "opening"
        elif any(word in section_lower for word in ["prepared", "remarks", "presentation"]):
            return "prepared_remarks"
        elif any(word in section_lower for word in ["q&a", "qa", "question", "answer"]):
            return "qa"
        elif any(word in section_lower for word in ["closing", "conclusion"]):
            return "closing"
        else:
            return section

    def calculate_importance_score(self, chunk: Dict) -> float:
        """
        Calculate importance score for chunk
        Based on section, speaker role, content characteristics
        """
        score = 0.5  # Base

        # Section importance
        section_scores = {
            'qa': 0.3,
            'prepared_remarks': 0.25,
            'opening': 0.1,
            'closing': 0.05
        }
        score += section_scores.get(chunk.get('section', ''), 0.0)

        # Speaker importance
        speaker_scores = {
            'CEO': 0.25,
            'CFO': 0.25,
            'Executive': 0.15,
            'IR': 0.05,
            'Analyst': 0.1,
            'Operator': 0.0
        }
        score += speaker_scores.get(chunk.get('speaker_role', ''), 0.0)

        # Data richness
        if chunk.get('contains_numbers'):
            score += 0.1

        # Forward-looking
        if chunk.get('forward_looking'):
            score += 0.1

        # Tonal confidence (if available)
        if 'tone_analysis' in chunk:
            confidence = chunk['tone_analysis'].get('confidence_level', 0.5)
            score += (confidence - 0.5) * 0.1

        return min(score, 1.0)

    def transform_to_finalized_schema(self, chunk: Dict, chunk_index: int, total_chunks: int) -> Dict:
        """
        Transform raw chunk to finalized schema
        Removes: topics, start_line, end_line
        Adds: company_name, parent_doc_id, source_type, fiscal_period, text_length, etc.
        """
        # Extract fiscal info
        ticker = chunk.get('ticker', 'TSLA')
        fiscal_year = chunk.get('fiscal_year', 2024)
        fiscal_quarter = chunk.get('fiscal_quarter', 'Q1')
        fiscal_period = f"{fiscal_year}{fiscal_quarter}"

        # Create parent_doc_id
        parent_doc_id = f"{ticker}_{fiscal_year}_{fiscal_quarter}_call"

        # Normalize section
        section = self.normalize_section(chunk.get('section', 'unknown'))

        # Normalize speaker role
        speaker = chunk.get('speaker', 'Unknown')
        current_role = chunk.get('speaker_role', 'Unknown')
        speaker_role, is_company_speaker = self.normalize_speaker_role(speaker, current_role)

        # Build finalized chunk
        finalized = {
            # === CORE IDENTIFIERS ===
            "chunk_id": chunk.get('chunk_id'),
            "parent_doc_id": parent_doc_id,
            "ticker": ticker,
            "company_name": "Tesla, Inc.",  # Hardcoded for prototype
            "source_type": "call",

            # === TEMPORAL ===
            "fiscal_year": fiscal_year,
            "fiscal_quarter": fiscal_quarter,
            "fiscal_period": fiscal_period,
            "call_date": chunk.get('call_date'),

            # === CONTENT ===
            "section": section,
            "text": chunk.get('text', ''),
            "text_length": len(chunk.get('text', '')),

            # === SPEAKER METADATA ===
            "speaker": speaker,
            "speaker_role": speaker_role,
            "is_company_speaker": is_company_speaker,

            # === ENRICHMENT ===
            "contains_numbers": chunk.get('contains_numbers', False),
            "forward_looking": chunk.get('forward_looking', False),
            "importance_score": None,  # Will calculate after tone analysis

            # === TONAL ANALYSIS (to be added by LLM) ===
            "tone_analysis": None,

            # === CONTEXT ===
            "chunk_metadata": {
                "chunk_index": chunk_index,
                "total_chunks": total_chunks,
                "prev_chunk_id": None if chunk_index == 0 else f"{ticker}_{fiscal_year}_{fiscal_quarter}_chunk_{chunk_index-1:03d}",
                "next_chunk_id": None if chunk_index == total_chunks - 1 else f"{ticker}_{fiscal_year}_{fiscal_quarter}_chunk_{chunk_index+1:03d}"
            },

            # === TIMESTAMPS ===
            "ingestion_timestamp": datetime.utcnow().isoformat() + "Z"
        }

        return finalized

    def process_chunk_with_tonal_analysis(self, chunk: Dict) -> Dict:
        """
        Apply LLM tonal analysis to a single chunk
        """
        if not self.use_llm or not self.tonal_analyzer:
            return chunk

        try:
            # Extract tonal analysis
            tonal_result = self.tonal_analyzer.analyze_earnings_call_chunk(
                text=chunk['text'],
                speaker=chunk.get('speaker'),
                speaker_role=chunk.get('speaker_role'),
                section=chunk.get('section')
            )

            # Add to chunk
            chunk['tone_analysis'] = {
                'sentiment': tonal_result.sentiment,
                'sentiment_score': tonal_result.sentiment_score,
                'confidence_level': tonal_result.confidence_level,
                'uncertainty_score': tonal_result.uncertainty_score,
                'forward_looking_score': tonal_result.forward_looking_score,
                'hedge_words_count': tonal_result.hedge_words_count,
                'certainty_words_count': tonal_result.certainty_words_count,
                'key_phrases': tonal_result.key_phrases
            }

            return chunk

        except Exception as e:
            print(f"  ✗ Tonal analysis failed: {e}")
            # Return chunk without tone_analysis
            return chunk

    def process_all_chunks(self, input_file: str, output_file: str) -> List[Dict]:
        """
        Process all earnings call chunks with LLM tonal analysis
        Transform to finalized schema
        """
        print(f"\n{'='*60}")
        print("EARNINGS CALL PROCESSOR - LLM Tonal Analysis")
        print(f"{'='*60}")

        # Load chunks
        if not Path(input_file).exists():
            print(f"❌ File not found: {input_file}")
            return []

        with open(input_file, 'r') as f:
            raw_chunks = json.load(f)

        print(f"Loaded {len(raw_chunks)} raw chunks")
        print(f"LLM Tonal Analysis: {'ENABLED' if self.use_llm else 'DISABLED'}")

        processed_chunks = []
        total_chunks = len(raw_chunks)

        print(f"\n{'='*60}")
        print("PROCESSING CHUNKS")
        print(f"{'='*60}")

        # Process in batches, save periodically
        batch_size = 50

        for i, raw_chunk in enumerate(raw_chunks):
            # Progress indicator every 10 chunks
            if (i + 1) % 10 == 0 or i == 0:
                print(f"Processing chunk {i+1}/{total_chunks}...")

            # Transform to finalized schema
            finalized_chunk = self.transform_to_finalized_schema(raw_chunk, i, total_chunks)

            # Apply LLM tonal analysis
            if self.use_llm:
                finalized_chunk = self.process_chunk_with_tonal_analysis(finalized_chunk)

            # Calculate importance score (after tone analysis)
            finalized_chunk['importance_score'] = self.calculate_importance_score(finalized_chunk)

            processed_chunks.append(finalized_chunk)

            # Save checkpoint every batch
            if (i + 1) % batch_size == 0:
                print(f"  ✓ Checkpoint: Processed {i+1}/{total_chunks} chunks")

        # Save results
        print(f"\n{'='*60}")
        print("SAVING RESULTS")
        print(f"{'='*60}")

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(processed_chunks, f, indent=2)

        print(f"Saved {len(processed_chunks)} processed chunks to:")
        print(f"  {output_file}")

        # Show sample results
        if processed_chunks and self.use_llm:
            print(f"\n{'='*60}")
            print("SAMPLE TONAL ANALYSIS RESULTS")
            print(f"{'='*60}")

            for i, chunk in enumerate(processed_chunks[:3]):
                if chunk.get('tone_analysis'):
                    print(f"\nChunk {i+1}:")
                    print(f"  Speaker: {chunk['speaker']} ({chunk['speaker_role']})")
                    print(f"  Section: {chunk['section']}")
                    print(f"  Text: {chunk['text'][:100]}...")
                    print(f"  Sentiment: {chunk['tone_analysis']['sentiment']} ({chunk['tone_analysis']['sentiment_score']:.2f})")
                    print(f"  Confidence: {chunk['tone_analysis']['confidence_level']:.2f}")
                    print(f"  Uncertainty: {chunk['tone_analysis']['uncertainty_score']:.2f}")
                    print(f"  Key Phrases: {chunk['tone_analysis']['key_phrases'][:3]}")

        # Statistics
        print(f"\n{'='*60}")
        print("STATISTICS")
        print(f"{'='*60}")

        if self.use_llm and any(c.get('tone_analysis') for c in processed_chunks):
            sentiments = [c['tone_analysis']['sentiment'] for c in processed_chunks if c.get('tone_analysis')]
            print(f"Sentiment Distribution:")
            print(f"  Bullish: {sentiments.count('bullish')}")
            print(f"  Neutral: {sentiments.count('neutral')}")
            print(f"  Bearish: {sentiments.count('bearish')}")

        sections = [c['section'] for c in processed_chunks]
        print(f"\nSection Distribution:")
        print(f"  Opening: {sections.count('opening')}")
        print(f"  Prepared Remarks: {sections.count('prepared_remarks')}")
        print(f"  Q&A: {sections.count('qa')}")
        print(f"  Closing: {sections.count('closing')}")

        speaker_roles = [c['speaker_role'] for c in processed_chunks]
        print(f"\nSpeaker Distribution:")
        for role in set(speaker_roles):
            print(f"  {role}: {speaker_roles.count(role)}")

        return processed_chunks


def main():
    """Main execution"""

    # Configuration
    input_file = "data/tesla/tesla_earning_calls/processed/all_tesla_calls_chunks.json"
    output_file = "data/tesla/tesla_earning_calls/processed/calls_finalized.json"

    # Check OpenAI key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found in .env file")
        print("Set use_llm=False to process without LLM analysis")
        return

    print("✅ OpenAI API key found")

    # Initialize processor with LLM enabled
    processor = EarningsCallProcessor(use_llm=True)

    # Process all chunks
    chunks = processor.process_all_chunks(input_file, output_file)

    print(f"\n✅ Successfully processed {len(chunks)} earnings call chunks")
    print(f"\n📁 Output: {output_file}")
    print(f"\n💡 Next steps:")
    print("1. Review the finalized chunks")
    print("2. Validate schema compliance with FINALIZED_CALL_SCHEMA.md")
    print("3. Generate embeddings for all three datasets")
    print("4. Ingest into OpenSearch")


if __name__ == "__main__":
    main()
