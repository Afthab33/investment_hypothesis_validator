"""
Production LLM analyzers using OpenAI GPT-4o-mini
Cost-effective and accurate for tonal analysis and signal extraction
"""

import os
import json
from openai import OpenAI
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import logging
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Configure OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

logger = logging.getLogger(__name__)


@dataclass
class TonalAnalysisResult:
    """Structured output from LLM tonal analysis"""
    sentiment: str  # bullish, neutral, bearish
    sentiment_score: float  # -1.0 to 1.0
    confidence_level: float  # 0.0 to 1.0
    uncertainty_score: float  # 0.0 to 1.0
    forward_looking_score: float  # 0.0 to 1.0
    hedge_words_count: int
    certainty_words_count: int
    key_phrases: List[str]
    reasoning: str  # LLM's explanation


@dataclass
class ChatSignal:
    """Extracted signal from chat message"""
    signal_type: str  # rumor, analysis, news, recommendation, general
    credibility_score: float  # 0.0 to 1.0
    actionability: str  # high, medium, low
    urgency: str  # immediate, short_term, long_term
    has_price_target: bool
    price_target: Optional[float]
    mentioned_tickers: List[str]
    confidence_in_extraction: float  # How confident the LLM is
    reasoning: str  # LLM's explanation


class OpenAITonalAnalyzer:
    """Tonal analyzer using OpenAI GPT-4o-mini"""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.max_retries = 3
        self.retry_delay = 2

    def analyze_earnings_call_chunk(self,
                                   text: str,
                                   speaker: str = None,
                                   speaker_role: str = None,
                                   section: str = None) -> TonalAnalysisResult:
        """
        Perform sophisticated tonal analysis using GPT-4o-mini
        Cost: ~$0.00015 per 1K tokens
        """

        # Build context
        context_parts = []
        if speaker:
            context_parts.append(f"Speaker: {speaker}")
        if speaker_role:
            context_parts.append(f"Role: {speaker_role}")
        if section:
            context_parts.append(f"Section: {section}")

        context = " | ".join(context_parts) if context_parts else "No additional context"

        system_prompt = """You are an expert financial analyst specializing in earnings call analysis.
Analyze the tone and sentiment of earnings call excerpts with nuance and context awareness.
Consider sarcasm, negation, hedging, and implicit meanings."""

        user_prompt = f"""Analyze this earnings call excerpt:

Context: {context}

Text: {text}

Provide analysis in this exact JSON format:
{{
    "sentiment": "bullish|neutral|bearish",
    "sentiment_score": -1.0 to 1.0,
    "confidence_level": 0.0 to 1.0,
    "uncertainty_score": 0.0 to 1.0,
    "forward_looking_score": 0.0 to 1.0,
    "hedge_words_count": integer,
    "certainty_words_count": integer,
    "key_phrases": ["phrase1", "phrase2", "phrase3"],
    "reasoning": "Brief explanation"
}}

Consider:
- Negation ("not growing" is bearish)
- Sarcasm ("fantastic... if you ignore losses")
- Hedged optimism vs genuine confidence
- Speaker authority (CEO statements matter more)"""

        for attempt in range(self.max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1,  # Low for consistency
                    max_tokens=500,
                    response_format={ "type": "json_object" }  # Force JSON response
                )

                result_text = response.choices[0].message.content
                return self._parse_tonal_response(result_text)

            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    return self._fallback_tonal_analysis()

    def _parse_tonal_response(self, response_text: str) -> TonalAnalysisResult:
        """Parse JSON response from GPT-4o-mini"""
        try:
            data = json.loads(response_text)
            return TonalAnalysisResult(
                sentiment=data.get('sentiment', 'neutral'),
                sentiment_score=float(data.get('sentiment_score', 0.0)),
                confidence_level=float(data.get('confidence_level', 0.5)),
                uncertainty_score=float(data.get('uncertainty_score', 0.5)),
                forward_looking_score=float(data.get('forward_looking_score', 0.0)),
                hedge_words_count=int(data.get('hedge_words_count', 0)),
                certainty_words_count=int(data.get('certainty_words_count', 0)),
                key_phrases=data.get('key_phrases', [])[:5],
                reasoning=data.get('reasoning', '')
            )
        except Exception as e:
            logger.error(f"Failed to parse response: {e}")
            return self._fallback_tonal_analysis()

    def _fallback_tonal_analysis(self) -> TonalAnalysisResult:
        """Fallback if LLM fails"""
        return TonalAnalysisResult(
            sentiment='neutral',
            sentiment_score=0.0,
            confidence_level=0.3,  # Low confidence for fallback
            uncertainty_score=0.5,
            forward_looking_score=0.0,
            hedge_words_count=0,
            certainty_words_count=0,
            key_phrases=[],
            reasoning="Fallback analysis (LLM unavailable)"
        )


class OpenAISignalExtractor:
    """Extract trading signals from chat using GPT-4o-mini"""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.max_retries = 3
        self.retry_delay = 2

    def extract_signal(self,
                       message: str,
                       trader_id: str = None,
                       trader_role: str = None,
                       context_messages: List[str] = None) -> ChatSignal:
        """
        Extract trading signal from Bloomberg chat message
        Cost: ~$0.00010 per message
        """

        # Build context
        context_parts = []
        if trader_id:
            context_parts.append(f"Trader: {trader_id}")
        if trader_role:
            context_parts.append(f"Role: {trader_role}")

        context = " | ".join(context_parts) if context_parts else ""

        # Add surrounding messages for context
        if context_messages:
            context += f"\n\nPrevious messages:\n" + "\n".join(context_messages[-3:])

        system_prompt = """You are analyzing Bloomberg IB chat messages for trading signals.
Extract actionable information and assess credibility.
Consider trader roles: PM > Analyst > Trader in credibility."""

        user_prompt = f"""Analyze this trading chat message:

{context}

Message: {message}

Provide analysis in this exact JSON format:
{{
    "signal_type": "rumor|analysis|news|recommendation|general",
    "credibility_score": 0.0 to 1.0,
    "actionability": "high|medium|low",
    "urgency": "immediate|short_term|long_term",
    "has_price_target": true/false,
    "price_target": null or number,
    "mentioned_tickers": ["TICKER1", "TICKER2"],
    "confidence_in_extraction": 0.0 to 1.0,
    "reasoning": "Brief explanation"
}}

Signal types:
- rumor: Unconfirmed ("hearing that...")
- analysis: Trader's interpretation
- news: Confirmed information
- recommendation: Buy/sell advice
- general: General discussion

Credibility factors:
- Specific numbers/dates = higher
- "Confirmed" = higher
- "Rumor/hearing" = lower
- PM role = +0.3 credibility"""

        for attempt in range(self.max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=400,
                    response_format={ "type": "json_object" }
                )

                result_text = response.choices[0].message.content
                return self._parse_signal_response(result_text)

            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    return self._fallback_signal_extraction()

    def _parse_signal_response(self, response_text: str) -> ChatSignal:
        """Parse JSON response"""
        try:
            data = json.loads(response_text)
            return ChatSignal(
                signal_type=data.get('signal_type', 'general'),
                credibility_score=float(data.get('credibility_score', 0.5)),
                actionability=data.get('actionability', 'low'),
                urgency=data.get('urgency', 'long_term'),
                has_price_target=data.get('has_price_target', False),
                price_target=data.get('price_target'),
                mentioned_tickers=data.get('mentioned_tickers', []),
                confidence_in_extraction=float(data.get('confidence_in_extraction', 0.5)),
                reasoning=data.get('reasoning', '')
            )
        except Exception as e:
            logger.error(f"Failed to parse response: {e}")
            return self._fallback_signal_extraction()

    def _fallback_signal_extraction(self) -> ChatSignal:
        """Fallback if LLM fails"""
        return ChatSignal(
            signal_type='general',
            credibility_score=0.3,
            actionability='low',
            urgency='long_term',
            has_price_target=False,
            price_target=None,
            mentioned_tickers=[],
            confidence_in_extraction=0.2,
            reasoning="Fallback extraction (LLM unavailable)"
        )


class BatchProcessor:
    """Process multiple chunks efficiently with rate limiting"""

    def __init__(self,
                 tonal_analyzer: OpenAITonalAnalyzer,
                 signal_extractor: OpenAISignalExtractor):
        self.tonal_analyzer = tonal_analyzer
        self.signal_extractor = signal_extractor
        self.rate_limit_delay = 0.1  # 100ms between requests

    def process_earnings_calls(self,
                              chunks: List[Dict],
                              save_path: str = None) -> List[Dict]:
        """Process earnings call chunks with tonal analysis"""

        processed = []
        total = len(chunks)

        print(f"Processing {total} earnings call chunks...")

        for i, chunk in enumerate(chunks):
            # Progress
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{total} chunks...")

            # Skip very short chunks
            if len(chunk.get('text', '')) < 50:
                chunk['tone_analysis'] = None
                processed.append(chunk)
                continue

            # Analyze tone
            result = self.tonal_analyzer.analyze_earnings_call_chunk(
                text=chunk['text'],
                speaker=chunk.get('speaker'),
                speaker_role=chunk.get('speaker_role'),
                section=chunk.get('section')
            )

            # Add to chunk
            chunk['tone_analysis'] = asdict(result)
            processed.append(chunk)

            # Rate limiting
            time.sleep(self.rate_limit_delay)

        # Save if path provided
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(processed, f, indent=2)
            print(f"Saved to {save_path}")

        return processed

    def process_chat_messages(self,
                             messages: List[Dict],
                             save_path: str = None) -> List[Dict]:
        """Process chat messages with signal extraction"""

        processed = []
        total = len(messages)

        print(f"Processing {total} chat messages...")

        for i, msg in enumerate(messages):
            # Progress
            if (i + 1) % 20 == 0:
                print(f"  Processed {i + 1}/{total} messages...")

            # Extract signal
            result = self.signal_extractor.extract_signal(
                message=msg['text'],
                trader_id=msg.get('trader_id'),
                trader_role=msg.get('trader_role')
            )

            # Add signal data
            msg.update(asdict(result))
            processed.append(msg)

            # Rate limiting
            time.sleep(self.rate_limit_delay)

        # Save if path provided
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(processed, f, indent=2)
            print(f"Saved to {save_path}")

        return processed