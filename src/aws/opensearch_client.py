"""
AWS OpenSearch Serverless client utilities.
Uses latest langchain patterns with AWS4Auth.
"""

import os
import boto3
from typing import Optional
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
from dotenv import load_dotenv

load_dotenv()


class OpenSearchClient:
    """Singleton client for AWS OpenSearch Serverless."""
    
    _instance = None
    _client = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self.region = os.getenv("AWS_REGION", "us-east-1")
            self.endpoint = os.getenv("OPENSEARCH_URL")
            self.index_name = os.getenv("OPENSEARCH_INDEX_NAME")
            
            if not self.endpoint:
                raise ValueError("OPENSEARCH_URL environment variable not set")
            
            # Remove https:// prefix if present
            self.host = self.endpoint.replace("https://", "").replace("http://", "")
            self._initialized = True
    
    def get_client(self) -> OpenSearch:
        """Get OpenSearch client with AWS4Auth."""
        if self._client is None:
            # Get AWS credentials
            credentials = boto3.Session(
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                region_name=self.region
            ).get_credentials()
            
            # Create AWS4Auth for OpenSearch Serverless
            awsauth = AWS4Auth(
                credentials.access_key,
                credentials.secret_key,
                self.region,
                'aoss',  # Service name for OpenSearch Serverless
                session_token=credentials.token
            )
            
            # Create OpenSearch client
            self._client = OpenSearch(
                hosts=[{'host': self.host, 'port': 443}],
                http_auth=awsauth,
                use_ssl=True,
                verify_certs=True,
                connection_class=RequestsHttpConnection,
                timeout=300
            )
        
        return self._client
    
    def get_index_name(self) -> str:
        """Get the configured index name."""
        return self.index_name


# Convenience function
def get_opensearch_client() -> OpenSearch:
    """Get OpenSearch client instance."""
    return OpenSearchClient().get_client()


def get_index_name() -> str:
    """Get the configured index name."""
    return OpenSearchClient().get_index_name()
