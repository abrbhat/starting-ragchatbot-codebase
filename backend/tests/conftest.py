"""Shared pytest fixtures for testing the RAG system."""

import pytest
from unittest.mock import Mock, MagicMock, patch
import sys
import os
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

# Mock chromadb, sentence_transformers, and anthropic BEFORE any imports
mock_chromadb = Mock()
mock_chromadb.config = Mock()
mock_chromadb.config.Settings = Mock()
mock_chromadb.PersistentClient = Mock()
mock_chromadb.utils = Mock()
mock_chromadb.utils.embedding_functions = Mock()
mock_chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction = Mock()

mock_sentence_transformers = Mock()
mock_sentence_transformers.SentenceTransformer = Mock()

mock_anthropic = Mock()
mock_anthropic.Anthropic = Mock()

sys.modules["chromadb"] = mock_chromadb
sys.modules["chromadb.config"] = mock_chromadb.config
sys.modules["sentence_transformers"] = mock_sentence_transformers
sys.modules["anthropic"] = mock_anthropic

# Add backend to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class SearchResults:
    """Local copy of SearchResults for testing without chromadb dependency."""

    documents: List[str]
    metadata: List[Dict[str, Any]]
    distances: List[float]
    error: Optional[str] = None

    def is_empty(self) -> bool:
        return len(self.documents) == 0


@pytest.fixture
def mock_vector_store():
    """Create a mock VectorStore for testing."""
    store = Mock()
    store.max_results = 5
    return store


@pytest.fixture
def sample_search_results():
    """Create sample SearchResults for testing."""
    return SearchResults(
        documents=[
            "MCP allows Claude to connect to external data sources and tools.",
            "The Model Context Protocol provides a standardized way for AI to access resources.",
        ],
        metadata=[
            {
                "course_title": "MCP: Build Rich-Context AI Apps",
                "lesson_number": 1,
                "chunk_index": 0,
            },
            {
                "course_title": "MCP: Build Rich-Context AI Apps",
                "lesson_number": 2,
                "chunk_index": 5,
            },
        ],
        distances=[0.25, 0.35],
    )


@pytest.fixture
def empty_search_results():
    """Create empty SearchResults for testing."""
    return SearchResults(documents=[], metadata=[], distances=[])


@pytest.fixture
def error_search_results():
    """Create SearchResults with an error."""
    return SearchResults(
        documents=[],
        metadata=[],
        distances=[],
        error="No course found matching 'NonExistentCourse'",
    )


@pytest.fixture
def sample_course_metadata():
    """Create sample course metadata dict as returned by VectorStore."""
    return {
        "title": "MCP: Build Rich-Context AI Apps with Anthropic",
        "course_link": "https://example.com/mcp-course",
        "instructor": "Test Instructor",
        "lessons": [
            {
                "lesson_number": 0,
                "lesson_title": "Introduction",
                "lesson_link": "https://example.com/lesson0",
            },
            {
                "lesson_number": 1,
                "lesson_title": "Why MCP",
                "lesson_link": "https://example.com/lesson1",
            },
            {
                "lesson_number": 2,
                "lesson_title": "MCP Architecture",
                "lesson_link": "https://example.com/lesson2",
            },
        ],
    }


@pytest.fixture
def course_search_tool(mock_vector_store):
    """Create a CourseSearchTool with mocked vector store."""
    from search_tools import CourseSearchTool

    tool = CourseSearchTool(mock_vector_store)
    return tool


@pytest.fixture
def course_outline_tool(mock_vector_store):
    """Create a CourseOutlineTool with mocked vector store."""
    from search_tools import CourseOutlineTool

    tool = CourseOutlineTool(mock_vector_store)
    return tool


@pytest.fixture
def tool_manager(course_search_tool, course_outline_tool):
    """Create a ToolManager with both tools registered."""
    from search_tools import ToolManager

    manager = ToolManager()
    manager.register_tool(course_search_tool)
    manager.register_tool(course_outline_tool)
    return manager
