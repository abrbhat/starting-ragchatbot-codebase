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


# ============================================================================
# API Testing Fixtures
# ============================================================================

@pytest.fixture
def mock_rag_system():
    """Create a mock RAGSystem for API testing."""
    mock_rag = Mock()
    mock_rag.query.return_value = ("This is a test response.", ["Source 1|https://example.com"])
    mock_rag.get_course_analytics.return_value = {
        "total_courses": 2,
        "course_titles": ["Course A", "Course B"]
    }
    mock_rag.session_manager = Mock()
    mock_rag.session_manager.create_session.return_value = "test-session-123"
    return mock_rag


@pytest.fixture
def test_app(mock_rag_system):
    """
    Create a test FastAPI app without static file mounting.

    This avoids the issue of missing frontend directory in test environment
    by defining API endpoints inline rather than importing from app.py.
    """
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    from typing import List, Optional

    app = FastAPI(title="Course Materials RAG System - Test")

    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None

    class QueryResponse(BaseModel):
        answer: str
        sources: List[str]
        session_id: str

    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]

    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = mock_rag_system.session_manager.create_session()

            answer, sources = mock_rag_system.query(request.query, session_id)

            return QueryResponse(
                answer=answer,
                sources=sources,
                session_id=session_id
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = mock_rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/")
    async def root():
        return {"message": "Course Materials RAG System API"}

    return app


@pytest.fixture
def test_client(test_app):
    """Create a TestClient for API testing."""
    from fastapi.testclient import TestClient
    return TestClient(test_app)


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
