"""
Improved Earnings Call Chunking Script for Investment Hypothesis Validator
Fixes speaker attribution, section identification, and Q&A pairing
"""

import re
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImprovedEarningsCallChunker:
    """
    Enhanced chunking that properly identifies speakers and preserves Q&A context
    """

    def __init__(
        self,
        chunk_size: int = 600,
        chunk_overlap: int = 50,  # Reduced overlap to minimize duplication
        min_chunk_size: int = 200
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

        # Known speaker mappings for Tesla calls
        self.known_speakers = {
            'elon musk': 'CEO',
            'vaibhav taneja': 'CFO',
            'lars moravy': 'VP Vehicle Engineering',
            'ashok elluswamy': 'Director Autopilot Software',
            'martin viecha': 'VP Investor Relations',
            'drew baglino': 'SVP Powertrain and Energy',
            'zachary kirkhorn': 'Former CFO'
        }

        # Analyst firms
        self.analyst_firms = [
            'morgan stanley', 'goldman sachs', 'bernstein', 'baird',
            'oppenheimer', 'wolfe research', 'piper sandler', 'canaccord'
        ]

        # Financial topics
        self.financial_topics = {
            'margins': ['margin', 'gross margin', 'operating margin'],
            'revenue': ['revenue', 'sales', 'top line'],
            'costs': ['cost', 'expense', 'capex', 'opex'],
            'guidance': ['guidance', 'outlook', 'forecast', 'expect'],
            'production': ['production', 'manufacturing', 'capacity'],
            'demand': ['demand', 'orders', 'deliveries', 'backlog'],
            'cash': ['cash', 'cash flow', 'free cash flow', 'liquidity'],
            'growth': ['growth', 'expansion', 'increase', 'yoy', 'year-over-year']
        }

        # Forward indicators
        self.forward_indicators = [
            'will', 'expect', 'anticipate', 'project', 'forecast',
            'outlook', 'guidance', 'target', 'goal', 'plan',
            'should', 'could', 'may', 'believe', 'intend'
        ]

    def chunk_transcript(
        self,
        file_path: str,
        ticker: str,
        company: str,
        fiscal_year: int,
        fiscal_quarter: str,
        call_date: str
    ) -> List[Dict]:
        """Enhanced chunking with better speaker identification"""

        logger.info(f"Processing {ticker} {fiscal_quarter} {fiscal_year} earnings call")

        # Read and parse transcript
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        doc_id = f"{ticker}_{fiscal_year}_{fiscal_quarter}_earnings_call"

        # Parse the transcript structure
        parsed_sections = self._parse_transcript_structure(content)

        # Process each section
        all_chunks = []
        chunk_index = 0

        for section_name, section_content in parsed_sections.items():
            logger.info(f"Processing section: {section_name}")

            if section_name == "qa":
                chunks = self._chunk_qa_enhanced(
                    section_content, ticker, company, fiscal_year,
                    fiscal_quarter, call_date, doc_id, file_path, chunk_index
                )
            else:
                chunks = self._chunk_prepared_remarks(
                    section_content, section_name, ticker, company,
                    fiscal_year, fiscal_quarter, call_date, doc_id,
                    file_path, chunk_index
                )

            all_chunks.extend(chunks)
            chunk_index += len(chunks)

        logger.info(f"Created {len(all_chunks)} chunks")
        return all_chunks

    def _parse_transcript_structure(self, content: str) -> Dict:
        """
        Better parsing that identifies sections and speakers
        """
        lines = content.split('\n')
        sections = {
            "opening": [],
            "qa": [],
            "closing": []
        }

        current_section = "opening"
        current_speaker = None
        current_speaker_content = []

        # Track line numbers
        line_num = 0

        for line in lines:
            line_num += 1
            line_stripped = line.strip()

            # Check for section transitions
            if 'question-and-answer' in line.lower() or 'q&a' in line.lower():
                # Save any pending speaker content
                if current_speaker and current_speaker_content:
                    sections[current_section].append({
                        'speaker': current_speaker['name'],
                        'role': current_speaker['role'],
                        'content': '\n'.join(current_speaker_content),
                        'start_line': current_speaker['start_line'],
                        'end_line': line_num - 1
                    })
                    current_speaker_content = []

                current_section = "qa"
                continue

            # Check for speaker changes
            potential_speaker = self._identify_speaker_enhanced(line_stripped, line_num)

            if potential_speaker:
                # Save previous speaker's content
                if current_speaker and current_speaker_content:
                    sections[current_section].append({
                        'speaker': current_speaker['name'],
                        'role': current_speaker['role'],
                        'content': '\n'.join(current_speaker_content),
                        'start_line': current_speaker['start_line'],
                        'end_line': line_num - 1
                    })
                    current_speaker_content = []

                current_speaker = potential_speaker

                # If the speaker name is on the same line as content, extract it
                if ':' in line:
                    content_after_colon = line.split(':', 1)[1].strip()
                    if content_after_colon:
                        current_speaker_content.append(content_after_colon)
            else:
                # Add line to current speaker's content
                if line_stripped:
                    current_speaker_content.append(line_stripped)

        # Save final speaker content
        if current_speaker and current_speaker_content:
            sections[current_section].append({
                'speaker': current_speaker['name'],
                'role': current_speaker['role'],
                'content': '\n'.join(current_speaker_content),
                'start_line': current_speaker['start_line'],
                'end_line': line_num
            })

        return sections

    def _identify_speaker_enhanced(self, line: str, line_num: int) -> Optional[Dict]:
        """
        Enhanced speaker identification
        """
        line_lower = line.lower()

        # Pattern 1: "A - Speaker Name" or "Q - Speaker Name"
        qa_pattern = r'^[AQ]\s*-\s*(.+?)$'
        qa_match = re.match(qa_pattern, line)
        if qa_match:
            name = qa_match.group(1).strip()
            role = self._determine_role(name)
            return {'name': name, 'role': role, 'start_line': line_num}

        # Pattern 2: Standalone name (e.g., "Elon Musk")
        if line and len(line.split()) <= 3 and line[0].isupper():
            # Check if it's a known speaker
            for known_name, known_role in self.known_speakers.items():
                if known_name in line_lower:
                    return {'name': line, 'role': known_role, 'start_line': line_num}

            # Check if next line has content (speaker pattern)
            if not any(char in line for char in [':', '.', ',', '?', '!']):
                return {'name': line, 'role': self._determine_role(line), 'start_line': line_num}

        return None

    def _determine_role(self, speaker_name: str) -> str:
        """
        Determine speaker's role based on name and context
        """
        name_lower = speaker_name.lower()

        # Check known speakers
        for known_name, role in self.known_speakers.items():
            if known_name in name_lower:
                return role

        # Check for analyst firms
        for firm in self.analyst_firms:
            if firm in name_lower:
                return f"Analyst - {firm.title()}"

        # Check for titles in name
        if 'ceo' in name_lower or 'chief executive' in name_lower:
            return 'CEO'
        elif 'cfo' in name_lower or 'chief financial' in name_lower:
            return 'CFO'
        elif 'operator' in name_lower:
            return 'Operator'

        # Default to Analyst for unknown speakers in Q&A
        return 'Analyst'

    def _chunk_qa_enhanced(
        self, qa_speakers: List[Dict], ticker: str, company: str,
        fiscal_year: int, fiscal_quarter: str, call_date: str,
        doc_id: str, file_path: str, start_chunk_index: int
    ) -> List[Dict]:
        """
        Enhanced Q&A chunking that preserves question-answer pairs
        """
        chunks = []
        chunk_index = start_chunk_index
        i = 0

        while i < len(qa_speakers):
            speaker_turn = qa_speakers[i]

            # Check if this is a question from an analyst
            if 'Analyst' in speaker_turn['role']:
                # Look for the answer in next turns
                answer_parts = []
                j = i + 1

                while j < len(qa_speakers) and 'Analyst' not in qa_speakers[j]['role']:
                    answer_parts.append(qa_speakers[j])
                    j += 1

                if answer_parts:
                    # Create Q&A pair chunk
                    qa_text = f"Question ({speaker_turn['speaker']}): {speaker_turn['content']}\n\n"

                    for answer in answer_parts:
                        qa_text += f"Answer ({answer['speaker']} - {answer['role']}): {answer['content']}\n\n"

                    # Check chunk size
                    if len(qa_text.split()) <= self.chunk_size * 1.5:  # Allow 50% larger for Q&A pairs
                        chunk = self._create_enhanced_chunk(
                            chunk_id=f"{ticker}_{fiscal_year}_{fiscal_quarter}_chunk_{chunk_index:03d}",
                            doc_id=doc_id,
                            source_file=file_path,
                            ticker=ticker,
                            company=company,
                            fiscal_year=fiscal_year,
                            fiscal_quarter=fiscal_quarter,
                            call_date=call_date,
                            chunk_index=chunk_index,
                            start_line=speaker_turn['start_line'],
                            end_line=answer_parts[-1]['end_line'] if answer_parts else speaker_turn['end_line'],
                            section="qa",
                            speaker=answer_parts[0]['speaker'] if answer_parts else speaker_turn['speaker'],
                            speaker_role=answer_parts[0]['role'] if answer_parts else speaker_turn['role'],
                            text=qa_text.strip(),
                            qa_metadata={
                                'questioner': speaker_turn['speaker'],
                                'answerers': [a['speaker'] for a in answer_parts]
                            }
                        )
                        chunks.append(chunk)
                        chunk_index += 1
                        i = j  # Skip processed answers
                        continue
                    else:
                        # Split if too long
                        # First chunk: question + first part of answer
                        chunk_text = f"Question ({speaker_turn['speaker']}): {speaker_turn['content']}\n\n"
                        if answer_parts:
                            chunk_text += f"Answer ({answer_parts[0]['speaker']} - {answer_parts[0]['role']}): {answer_parts[0]['content'][:500]}..."

                        chunk = self._create_enhanced_chunk(
                            chunk_id=f"{ticker}_{fiscal_year}_{fiscal_quarter}_chunk_{chunk_index:03d}",
                            doc_id=doc_id,
                            source_file=file_path,
                            ticker=ticker,
                            company=company,
                            fiscal_year=fiscal_year,
                            fiscal_quarter=fiscal_quarter,
                            call_date=call_date,
                            chunk_index=chunk_index,
                            start_line=speaker_turn['start_line'],
                            end_line=speaker_turn['end_line'],
                            section="qa",
                            speaker=answer_parts[0]['speaker'] if answer_parts else speaker_turn['speaker'],
                            speaker_role=answer_parts[0]['role'] if answer_parts else speaker_turn['role'],
                            text=chunk_text,
                            qa_metadata={
                                'questioner': speaker_turn['speaker'],
                                'is_partial': True
                            }
                        )
                        chunks.append(chunk)
                        chunk_index += 1

                        # Continue with answer chunks
                        for answer in answer_parts:
                            chunk = self._create_enhanced_chunk(
                                chunk_id=f"{ticker}_{fiscal_year}_{fiscal_quarter}_chunk_{chunk_index:03d}",
                                doc_id=doc_id,
                                source_file=file_path,
                                ticker=ticker,
                                company=company,
                                fiscal_year=fiscal_year,
                                fiscal_quarter=fiscal_quarter,
                                call_date=call_date,
                                chunk_index=chunk_index,
                                start_line=answer['start_line'],
                                end_line=answer['end_line'],
                                section="qa",
                                speaker=answer['speaker'],
                                speaker_role=answer['role'],
                                text=answer['content'],
                                qa_metadata={
                                    'continuation_of_question': speaker_turn['speaker']
                                }
                            )
                            chunks.append(chunk)
                            chunk_index += 1

                        i = j
                        continue

            # Regular speaker turn (not Q&A pair)
            if len(speaker_turn['content'].split()) >= self.min_chunk_size:
                chunk = self._create_enhanced_chunk(
                    chunk_id=f"{ticker}_{fiscal_year}_{fiscal_quarter}_chunk_{chunk_index:03d}",
                    doc_id=doc_id,
                    source_file=file_path,
                    ticker=ticker,
                    company=company,
                    fiscal_year=fiscal_year,
                    fiscal_quarter=fiscal_quarter,
                    call_date=call_date,
                    chunk_index=chunk_index,
                    start_line=speaker_turn['start_line'],
                    end_line=speaker_turn['end_line'],
                    section="qa",
                    speaker=speaker_turn['speaker'],
                    speaker_role=speaker_turn['role'],
                    text=speaker_turn['content']
                )
                chunks.append(chunk)
                chunk_index += 1

            i += 1

        return chunks

    def _chunk_prepared_remarks(
        self, speakers: List[Dict], section_name: str, ticker: str,
        company: str, fiscal_year: int, fiscal_quarter: str,
        call_date: str, doc_id: str, file_path: str,
        start_chunk_index: int
    ) -> List[Dict]:
        """
        Chunk opening/closing remarks
        """
        chunks = []
        chunk_index = start_chunk_index

        for speaker_turn in speakers:
            # Split long remarks into chunks
            content = speaker_turn['content']
            words = content.split()

            if len(words) <= self.chunk_size:
                # Single chunk
                chunk = self._create_enhanced_chunk(
                    chunk_id=f"{ticker}_{fiscal_year}_{fiscal_quarter}_chunk_{chunk_index:03d}",
                    doc_id=doc_id,
                    source_file=file_path,
                    ticker=ticker,
                    company=company,
                    fiscal_year=fiscal_year,
                    fiscal_quarter=fiscal_quarter,
                    call_date=call_date,
                    chunk_index=chunk_index,
                    start_line=speaker_turn['start_line'],
                    end_line=speaker_turn['end_line'],
                    section=section_name,
                    speaker=speaker_turn['speaker'],
                    speaker_role=speaker_turn['role'],
                    text=content
                )
                chunks.append(chunk)
                chunk_index += 1
            else:
                # Split into multiple chunks
                current_pos = 0
                while current_pos < len(words):
                    chunk_words = words[current_pos:current_pos + self.chunk_size]

                    # Add overlap from previous chunk if not first chunk
                    if current_pos > 0 and self.chunk_overlap > 0:
                        overlap_start = max(0, current_pos - self.chunk_overlap)
                        overlap_words = words[overlap_start:current_pos]
                        chunk_words = overlap_words + chunk_words

                    chunk_text = ' '.join(chunk_words)

                    chunk = self._create_enhanced_chunk(
                        chunk_id=f"{ticker}_{fiscal_year}_{fiscal_quarter}_chunk_{chunk_index:03d}",
                        doc_id=doc_id,
                        source_file=file_path,
                        ticker=ticker,
                        company=company,
                        fiscal_year=fiscal_year,
                        fiscal_quarter=fiscal_quarter,
                        call_date=call_date,
                        chunk_index=chunk_index,
                        start_line=speaker_turn['start_line'],
                        end_line=speaker_turn['end_line'],
                        section=section_name,
                        speaker=speaker_turn['speaker'],
                        speaker_role=speaker_turn['role'],
                        text=chunk_text
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                    current_pos += self.chunk_size

        return chunks

    def _create_enhanced_chunk(
        self, chunk_id: str, doc_id: str, source_file: str,
        ticker: str, company: str, fiscal_year: int,
        fiscal_quarter: str, call_date: str, chunk_index: int,
        start_line: int, end_line: int, section: str,
        speaker: str, speaker_role: str, text: str,
        qa_metadata: Dict = None
    ) -> Dict:
        """
        Create chunk with enhanced metadata
        """
        # Extract topics
        topics = self._extract_topics(text)

        # Check for numbers and forward-looking
        contains_numbers = self._contains_financial_numbers(text)
        forward_looking = self._is_forward_looking(text)

        chunk = {
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "source_file": source_file,
            "ticker": ticker,
            "company": company,
            "fiscal_year": fiscal_year,
            "fiscal_quarter": fiscal_quarter,
            "call_date": call_date,
            "chunk_index": chunk_index,
            "start_line": start_line,
            "end_line": end_line,
            "section": section,
            "speaker": speaker,
            "speaker_role": speaker_role,
            "topics": topics,
            "contains_numbers": contains_numbers,
            "forward_looking": forward_looking,
            "text": text
        }

        # Add Q&A metadata if present
        if qa_metadata:
            chunk["qa_metadata"] = qa_metadata

        return chunk

    def _extract_topics(self, text: str) -> List[str]:
        """Extract relevant financial topics"""
        text_lower = text.lower()
        found_topics = []

        for topic, keywords in self.financial_topics.items():
            if any(keyword in text_lower for keyword in keywords):
                found_topics.append(topic)

        return found_topics

    def _contains_financial_numbers(self, text: str) -> bool:
        """Check if text contains financial numbers"""
        patterns = [
            r'\d+\.?\d*\s*%',  # Percentages
            r'\$\s*\d+',        # Dollar amounts
            r'\d+\s*(million|billion|thousand)',
            r'Q[1-4]\s*\d{4}',  # Quarter references
        ]

        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def _is_forward_looking(self, text: str) -> bool:
        """Check for forward-looking statements"""
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in self.forward_indicators)

    def save_chunks(self, chunks: List[Dict], output_file: str):
        """Save chunks to JSON"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(chunks)} chunks to {output_file}")


def main():
    """Test with Tesla Q1 2024"""
    chunker = ImprovedEarningsCallChunker(
        chunk_size=600,
        chunk_overlap=50,
        min_chunk_size=200
    )

    chunks = chunker.chunk_transcript(
        file_path="data/calls/tesla_q1_2024.txt",
        ticker="TSLA",
        company="Tesla Inc",
        fiscal_year=2024,
        fiscal_quarter="Q1",
        call_date="2024-04-23"
    )

    chunker.save_chunks(
        chunks,
        "data/processed/chunks/tesla_q1_2024_chunks_v2.json"
    )

    # Print statistics
    print(f"\nImproved Chunking Statistics:")
    print(f"Total chunks: {len(chunks)}")

    # Check speaker attribution
    speakers = {}
    for chunk in chunks:
        speaker = f"{chunk['speaker']} ({chunk['speaker_role']})"
        speakers[speaker] = speakers.get(speaker, 0) + 1

    print(f"\nSpeaker distribution:")
    for speaker, count in sorted(speakers.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {speaker}: {count} chunks")

    # Section distribution
    sections = {}
    for chunk in chunks:
        sections[chunk['section']] = sections.get(chunk['section'], 0) + 1

    print(f"\nSection distribution:")
    for section, count in sections.items():
        print(f"  {section}: {count} chunks")

    # Sample improved chunk
    print(f"\nSample improved chunk:")
    print(json.dumps(chunks[0], indent=2))


if __name__ == "__main__":
    main()