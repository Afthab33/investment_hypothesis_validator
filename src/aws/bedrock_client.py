"""
AWS Bedrock client utilities for LLM and embeddings.
Uses latest langchain_aws patterns with retry logic.
"""

import os
import time
import logging
from typing import Optional, Any
from functools import wraps
import boto3
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain_core.messages import BaseMessage
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


def retry_with_backoff(max_retries: int = 5, initial_delay: float = 1.0):
    """
    Decorator for exponential backoff retry logic.

    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    error_msg = str(e)

                    # Check if it's a throttling error
                    if "ThrottlingException" in error_msg or "Too many requests" in error_msg:
                        if attempt < max_retries - 1:
                            logger.warning(
                                f"Throttling detected, retrying in {delay:.1f}s "
                                f"(attempt {attempt + 1}/{max_retries})"
                            )
                            time.sleep(delay)
                            delay *= 2  # Exponential backoff
                            continue

                    # For other errors, raise immediately
                    raise

            # All retries exhausted
            logger.error(f"Max retries ({max_retries}) exhausted")
            raise last_exception

        return wrapper
    return decorator


class BedrockClient:
    """Singleton client for AWS Bedrock services with retry logic."""

    _instance = None
    _embeddings = None
    _llm = None
    _last_llm_call = 0
    _min_delay_between_calls = 2.0  # 2s between calls to avoid throttling

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, '_initialized'):
            self.region = os.getenv("BEDROCK_REGION", "us-east-1")
            self.embedding_model_id = os.getenv(
                "BEDROCK_EMBEDDING_MODEL",
                "amazon.titan-embed-text-v2:0"
            )
            self.llm_model_id = os.getenv(
                "BEDROCK_LLM_MODEL",
                "anthropic.claude-3-5-sonnet-20240620-v1:0"
            )
            self._initialized = True
    
    def get_embeddings(self) -> BedrockEmbeddings:
        """Get Bedrock embeddings instance."""
        if self._embeddings is None:
            self._embeddings = BedrockEmbeddings(
                model_id=self.embedding_model_id,
                region_name=self.region
            )
        return self._embeddings
    
    def get_llm(
        self,
        temperature: float = 0,
        max_tokens: Optional[int] = None
    ) -> ChatBedrock:
        """Get Bedrock LLM instance."""
        kwargs = {
            "model_id": self.llm_model_id,
            "region_name": self.region,
            "temperature": temperature,
        }

        if max_tokens:
            kwargs["model_kwargs"] = {"max_tokens": max_tokens}

        return ChatBedrock(**kwargs)

    @retry_with_backoff(max_retries=5, initial_delay=1.0)
    def invoke_llm_with_retry(
        self,
        messages: list,
        temperature: float = 0,
        max_tokens: Optional[int] = None
    ) -> Any:
        """
        Invoke LLM with retry logic and rate limiting.

        Args:
            messages: List of messages to send
            temperature: LLM temperature
            max_tokens: Max tokens to generate

        Returns:
            LLM response
        """
        # Rate limiting: ensure minimum delay between calls
        current_time = time.time()
        time_since_last_call = current_time - self._last_llm_call
        if time_since_last_call < self._min_delay_between_calls:
            sleep_time = self._min_delay_between_calls - time_since_last_call
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f}s")
            time.sleep(sleep_time)

        # Make the call
        llm = self.get_llm(temperature=temperature, max_tokens=max_tokens)
        response = llm.invoke(messages)

        # Update last call time
        self._last_llm_call = time.time()

        return response


# Convenience functions
def get_embeddings() -> BedrockEmbeddings:
    """Get Bedrock embeddings instance."""
    return BedrockClient().get_embeddings()


def get_llm(temperature: float = 0, max_tokens: Optional[int] = None) -> ChatBedrock:
    """Get Bedrock LLM instance."""
    return BedrockClient().get_llm(temperature, max_tokens)


def invoke_llm_with_retry(
    messages: list,
    temperature: float = 0,
    max_tokens: Optional[int] = None
) -> Any:
    """
    Invoke LLM with retry logic and rate limiting.

    Args:
        messages: List of messages to send
        temperature: LLM temperature
        max_tokens: Max tokens to generate

    Returns:
        LLM response
    """
    return BedrockClient().invoke_llm_with_retry(messages, temperature, max_tokens)