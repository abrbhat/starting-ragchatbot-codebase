"""Tests for RAGSystem content-query handling."""
import pytest
from unittest.mock import Mock, MagicMock, patch
import sys
import os
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class SearchResults:
    """Local SearchResults for testing without chromadb dependency."""
    documents: List[str]
    metadata: List[Dict[str, Any]]
    distances: List[float]
    error: Optional[str] = None

    def is_empty(self) -> bool:
        return len(self.documents) == 0


class TestRAGSystemContentQueries:
    """Test suite for RAG system handling of content-related queries."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock config object."""
        config = Mock()
        config.CHUNK_SIZE = 800
        config.CHUNK_OVERLAP = 100
        config.CHROMA_PATH = "./test_chroma"
        config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        config.MAX_RESULTS = 5
        config.ANTHROPIC_API_KEY = "test-key"
        config.ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
        config.MAX_HISTORY = 10
        return config

    @pytest.fixture
    def mock_rag_system(self, mock_config):
        """Create a RAGSystem with mocked components."""
        with patch('rag_system.DocumentProcessor'), \
             patch('rag_system.VectorStore') as MockVectorStore, \
             patch('rag_system.AIGenerator') as MockAIGenerator, \
             patch('rag_system.SessionManager'):

            from rag_system import RAGSystem

            # Configure mock vector store
            mock_vector_store = MockVectorStore.return_value
            mock_vector_store.max_results = 5
            mock_vector_store.search.return_value = SearchResults(
                documents=["Content about MCP protocol"],
                metadata=[{"course_title": "MCP Course", "lesson_number": 1}],
                distances=[0.2]
            )
            mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson1"
            mock_vector_store.get_course_metadata.return_value = {
                "title": "MCP Course",
                "course_link": "https://example.com",
                "lessons": [{"lesson_number": 1, "lesson_title": "Intro"}]
            }

            # Configure mock AI generator
            mock_ai_generator = MockAIGenerator.return_value
            mock_ai_generator.generate_response.return_value = "MCP is a protocol..."

            rag_system = RAGSystem(mock_config)
            rag_system._mock_vector_store = mock_vector_store
            rag_system._mock_ai_generator = mock_ai_generator

            return rag_system

    def test_query_passes_tools_to_ai_generator(self, mock_rag_system):
        """Test that query method passes tool definitions to AIGenerator."""
        mock_rag_system.query("What is MCP?")

        # Verify AI generator was called with tools
        call_kwargs = mock_rag_system.ai_generator.generate_response.call_args[1]
        assert "tools" in call_kwargs
        assert call_kwargs["tools"] is not None
        assert len(call_kwargs["tools"]) >= 1

    def test_query_passes_tool_manager(self, mock_rag_system):
        """Test that query method passes tool_manager to AIGenerator."""
        mock_rag_system.query("What is MCP?")

        call_kwargs = mock_rag_system.ai_generator.generate_response.call_args[1]
        assert "tool_manager" in call_kwargs
        assert call_kwargs["tool_manager"] is not None

    def test_query_returns_response_and_sources(self, mock_rag_system):
        """Test that query returns tuple of (response, sources)."""
        response, sources = mock_rag_system.query("What is MCP?")

        assert isinstance(response, str)
        assert isinstance(sources, list)

    def test_query_retrieves_sources_from_tool_manager(self, mock_rag_system):
        """Test that sources are retrieved from tool manager after query."""
        # Simulate tool having captured sources
        mock_rag_system.search_tool.last_sources = ["MCP Course - Lesson 1|https://example.com"]

        response, sources = mock_rag_system.query("What is MCP?")

        assert len(sources) > 0

    def test_query_resets_sources_after_retrieval(self, mock_rag_system):
        """Test that sources are reset after being retrieved."""
        mock_rag_system.search_tool.last_sources = ["Source 1"]

        mock_rag_system.query("What is MCP?")

        # After query, sources should be reset
        assert mock_rag_system.search_tool.last_sources == []

    def test_query_includes_session_history_when_provided(self, mock_rag_system):
        """Test that conversation history is passed when session_id provided."""
        mock_rag_system.session_manager.get_conversation_history.return_value = "Previous conversation"

        mock_rag_system.query("Follow up question", session_id="session123")

        call_kwargs = mock_rag_system.ai_generator.generate_response.call_args[1]
        assert call_kwargs["conversation_history"] == "Previous conversation"

    def test_query_adds_exchange_to_session_history(self, mock_rag_system):
        """Test that query adds exchange to session history."""
        mock_rag_system.ai_generator.generate_response.return_value = "Response text"

        mock_rag_system.query("User question", session_id="session123")

        mock_rag_system.session_manager.add_exchange.assert_called_once()
        call_args = mock_rag_system.session_manager.add_exchange.call_args[0]
        assert call_args[0] == "session123"
        assert "User question" in call_args[1]


class TestRAGSystemToolRegistration:
    """Test suite for RAGSystem tool registration."""

    @pytest.fixture
    def mock_rag_system(self):
        """Create RAGSystem with patches for tool registration testing."""
        mock_config = Mock()
        mock_config.CHUNK_SIZE = 800
        mock_config.CHUNK_OVERLAP = 100
        mock_config.CHROMA_PATH = "./test_chroma"
        mock_config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        mock_config.MAX_RESULTS = 5
        mock_config.ANTHROPIC_API_KEY = "test-key"
        mock_config.ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
        mock_config.MAX_HISTORY = 10

        with patch('rag_system.DocumentProcessor'), \
             patch('rag_system.VectorStore'), \
             patch('rag_system.AIGenerator'), \
             patch('rag_system.SessionManager'):

            from rag_system import RAGSystem
            return RAGSystem(mock_config)

    def test_search_tool_is_registered(self, mock_rag_system):
        """Test that CourseSearchTool is registered in tool manager."""
        assert "search_course_content" in mock_rag_system.tool_manager.tools

    def test_outline_tool_is_registered(self, mock_rag_system):
        """Test that CourseOutlineTool is registered in tool manager."""
        assert "get_course_outline" in mock_rag_system.tool_manager.tools

    def test_tool_definitions_available(self, mock_rag_system):
        """Test that tool definitions can be retrieved."""
        definitions = mock_rag_system.tool_manager.get_tool_definitions()

        assert len(definitions) >= 2
        tool_names = [d["name"] for d in definitions]
        assert "search_course_content" in tool_names
        assert "get_course_outline" in tool_names


class TestRAGSystemQueryPrompt:
    """Test suite for RAGSystem query prompt formatting."""

    @pytest.fixture
    def mock_rag_system(self):
        """Create RAGSystem for prompt testing."""
        mock_config = Mock()
        mock_config.CHUNK_SIZE = 800
        mock_config.CHUNK_OVERLAP = 100
        mock_config.CHROMA_PATH = "./test_chroma"
        mock_config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        mock_config.MAX_RESULTS = 5
        mock_config.ANTHROPIC_API_KEY = "test-key"
        mock_config.ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
        mock_config.MAX_HISTORY = 10

        with patch('rag_system.DocumentProcessor'), \
             patch('rag_system.VectorStore'), \
             patch('rag_system.AIGenerator'), \
             patch('rag_system.SessionManager'):

            from rag_system import RAGSystem
            return RAGSystem(mock_config)

    def test_query_wraps_user_question(self, mock_rag_system):
        """Test that user query is wrapped with instruction context."""
        mock_rag_system.ai_generator.generate_response.return_value = "Response"

        mock_rag_system.query("What is MCP?")

        call_args = mock_rag_system.ai_generator.generate_response.call_args
        query_param = call_args[1]["query"]

        # Should contain the original question
        assert "What is MCP?" in query_param
        # Should have some context about course materials
        assert "course" in query_param.lower()


class TestRAGSystemEndToEnd:
    """End-to-end tests for content query flow."""

    @pytest.fixture
    def integrated_rag_system(self):
        """Create RAGSystem with real tool integration but mocked externals."""
        mock_config = Mock()
        mock_config.CHUNK_SIZE = 800
        mock_config.CHUNK_OVERLAP = 100
        mock_config.CHROMA_PATH = "./test_chroma"
        mock_config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        mock_config.MAX_RESULTS = 5
        mock_config.ANTHROPIC_API_KEY = "test-key"
        mock_config.ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
        mock_config.MAX_HISTORY = 10

        with patch('rag_system.DocumentProcessor'), \
             patch('rag_system.VectorStore') as MockVectorStore, \
             patch('rag_system.AIGenerator') as MockAIGenerator, \
             patch('rag_system.SessionManager'):

            # Set up vector store to return results
            mock_vs = MockVectorStore.return_value
            mock_vs.max_results = 5
            mock_vs.search.return_value = SearchResults(
                documents=["MCP is the Model Context Protocol..."],
                metadata=[{"course_title": "MCP Course", "lesson_number": 1}],
                distances=[0.15]
            )
            mock_vs.get_lesson_link.return_value = "https://example.com/mcp/lesson1"

            from rag_system import RAGSystem
            rag = RAGSystem(mock_config)

            # Store mock for assertions
            rag._mock_vector_store = mock_vs
            return rag

    def test_content_query_triggers_vector_search(self, integrated_rag_system):
        """Test that content queries can trigger vector store search."""
        # Execute search tool directly to verify it works
        result = integrated_rag_system.tool_manager.execute_tool(
            "search_course_content",
            query="What is MCP?"
        )

        # Verify vector store was searched
        integrated_rag_system._mock_vector_store.search.assert_called()
        assert isinstance(result, str)

    def test_search_results_formatted_with_sources(self, integrated_rag_system):
        """Test that search results include source information."""
        result = integrated_rag_system.tool_manager.execute_tool(
            "search_course_content",
            query="MCP protocol"
        )

        # Result should include course context
        assert "MCP Course" in result or "MCP is" in result

        # Sources should be tracked
        sources = integrated_rag_system.tool_manager.get_last_sources()
        assert len(sources) > 0

    def test_search_with_course_filter(self, integrated_rag_system):
        """Test that course filter is passed to vector store."""
        integrated_rag_system.tool_manager.execute_tool(
            "search_course_content",
            query="architecture",
            course_name="MCP"
        )

        # Verify filter was passed
        call_kwargs = integrated_rag_system._mock_vector_store.search.call_args[1]
        assert call_kwargs["course_name"] == "MCP"

    def test_search_with_lesson_filter(self, integrated_rag_system):
        """Test that lesson filter is passed to vector store."""
        integrated_rag_system.tool_manager.execute_tool(
            "search_course_content",
            query="introduction",
            lesson_number=0
        )

        call_kwargs = integrated_rag_system._mock_vector_store.search.call_args[1]
        assert call_kwargs["lesson_number"] == 0
