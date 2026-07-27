"""
AinSeba - Configuration Management
Centralized settings for the entire project.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()

# ============================================
# Path Configuration
# ============================================
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA_DIR = PROJECT_ROOT / os.getenv("RAW_DATA_DIR", "data/raw")
PROCESSED_DATA_DIR = PROJECT_ROOT / os.getenv("PROCESSED_DATA_DIR", "data/processed")

# Ensure directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ============================================
# Chunking Configuration
# ============================================
CHUNK_SIZE_TOKENS = int(os.getenv("CHUNK_SIZE_TOKENS", 600))
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", 100))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# ============================================
# Logging Configuration
# ============================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ============================================
# Phase 2: Vector Store & Retrieval Config
# ============================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
CHROMA_PERSIST_DIR = PROJECT_ROOT / os.getenv("CHROMA_PERSIST_DIR", "chroma_db")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "ainseba_laws")

RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", 10))
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", 5))
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# Load the cross-encoder reranker at all. Keep this true locally: reranking
# measurably improves ordering (it scored Labour Act s.100 at 3.381 while
# pushing a superficially similar adolescent-hours clause to -1.219).
# Set false on memory-capped hosts -- torch plus the model needs well over the
# 512MB that free-tier PaaS instances provide. Retrieval then falls back to
# pure vector similarity.
USE_RERANKER = os.getenv("USE_RERANKER", "true").strip().lower() in ("1", "true", "yes", "on")

# ============================================
# Phase 3: RAG Chain & LLM Configuration
# ============================================
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.1))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", 1500))
CONVERSATION_MEMORY_K = int(os.getenv("CONVERSATION_MEMORY_K", 5))

# ============================================
# Phase 4: Bilingual Support Configuration
# ============================================
TRANSLATION_MODEL = os.getenv("TRANSLATION_MODEL", "gpt-4o-mini")
DEFAULT_RESPONSE_LANGUAGE = os.getenv("DEFAULT_RESPONSE_LANGUAGE", "auto")  # auto, en, bn

# ============================================
# Phase 5: FastAPI Backend Configuration
# ============================================
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
API_RATE_LIMIT = int(os.getenv("API_RATE_LIMIT", 30))
API_RATE_WINDOW = int(os.getenv("API_RATE_WINDOW", 60))
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

# ============================================
# Law Document Registry
# Each entry contains metadata about a law PDF.
# This registry drives the entire ingestion pipeline.
# ============================================
LAW_REGISTRY = [
    {
        "id": "labour_act_2006",
        "name": "Bangladesh Labour Act 2006",
        "filename": "bangladesh_labour_act_2006.pdf",
        "source_url": "http://bdlaws.minlaw.gov.bd/act-952.html",
        "priority": "P0",
        "category": "Employment",
        "year": 2006,
        "language": "english",
    },
    {
        "id": "penal_code_1860",
        "name": "The Penal Code 1860",
        "filename": "penal_code_1860.pdf",
        "source_url": "http://bdlaws.minlaw.gov.bd/act-11.html",
        "priority": "P0",
        "category": "Criminal Law",
        "year": 1860,
        "language": "english",
    },
    {
        "id": "consumer_rights_2009",
        "name": "Consumer Rights Protection Act 2009",
        "filename": "consumer_rights_protection_act_2009.pdf",
        "source_url": "http://bdlaws.minlaw.gov.bd/act-1035.html",
        "priority": "P0",
        "category": "Consumer Rights",
        "year": 2009,
        "language": "english",
    },
    {
        "id": "cyber_security_2023",
        "name": "Cyber Security Act 2023",
        "filename": "cyber_security_act_2023.pdf",
        "source_url": "http://bdlaws.minlaw.gov.bd/act-details-1470.html",
        "priority": "P0",
        "category": "Cyber Law",
        "year": 2023,
        "language": "english",
    },
    {
        "id": "rent_control_1991",
        "name": "The Rent Control Act 1991",
        "filename": "rent_control_act_1991.pdf",
        "source_url": "http://bdlaws.minlaw.gov.bd/act-786.html",
        "priority": "P1",
        "category": "Property",
        "year": 1991,
        "language": "english",
    },
    {
        "id": "muslim_family_law_1961",
        "name": "Muslim Family Laws Ordinance 1961",
        "filename": "muslim_family_laws_ordinance_1961.pdf",
        "source_url": "http://bdlaws.minlaw.gov.bd/act-305.html",
        "priority": "P1",
        "category": "Family Law",
        "year": 1961,
        "language": "english",
    },
    {
        "id": "companies_act_1994",
        "name": "The Companies Act 1994",
        "filename": "companies_act_1994.pdf",
        "source_url": "http://bdlaws.minlaw.gov.bd/act-788.html",
        "priority": "P1",
        "category": "Business",
        "year": 1994,
        "language": "english",
    },
    {
        "id": "constitution_bd",
        "name": "The Constitution of the People's Republic of Bangladesh",
        "filename": "constitution_of_bangladesh.pdf",
        "source_url": "http://bdlaws.minlaw.gov.bd/act-367.html",
        "priority": "P2",
        "category": "Constitutional Law",
        "year": 1972,
        "language": "english",
    },
    {
        "id": "environment_act_1995",
        "name": "Bangladesh Environment Conservation Act 1995",
        "filename": "environment_conservation_act_1995.pdf",
        "source_url": "http://bdlaws.minlaw.gov.bd/act-805.html",
        "priority": "P1",
        "category": "Environmental Law",
        "year": 1995,
        "language": "english",
    },
    {
        "id": "tenancy_act_1950",
        "name": "State Acquisition and Tenancy Act 1950",
        "filename": "state_acquisition_tenancy_act_1950.pdf",
        "source_url": "http://bdlaws.minlaw.gov.bd/act-250.html",
        "priority": "P1",
        "category": "Property Law",
        "year": 1950,
        "language": "english",
    },
]


def get_law_by_id(law_id: str) -> dict | None:
    """Look up a law entry by its ID."""
    for law in LAW_REGISTRY:
        if law["id"] == law_id:
            return law
    return None


def get_laws_by_priority(priority: str) -> list[dict]:
    """Get all laws matching a priority level (P0, P1, P2)."""
    return [law for law in LAW_REGISTRY if law["priority"] == priority]