"""Tests for AIGenerator tool calling behavior."""
import pytest
from unittest.mock import Mock, MagicMock, patch
import sys
import os
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import after conftest has mocked chromadb
from ai_generator import AIGenerator
from search_tools import CourseSearchTool, ToolManager


@dataclass
class SearchResults:
    """Local SearchResults for testing without chromadb dependency."""
    documents: List[str]
    metadata: List[Dict[str, Any]]
    distances: List[float]
    error: Optional[str] = None

    def is_empty(self) -> bool:
        return len(self.documents) == 0


class TestAIGeneratorToolCalling:
    """Test suite for AIGenerator's tool calling mechanism."""

    @pytest.fixture
    def mock_anthropic_client(self):
        """Create a mock Anthropic client."""
        with patch('ai_generator.anthropic.Anthropic') as mock:
            yield mock

    @pytest.fixture
    def ai_generator(self, mock_anthropic_client):
        """Create an AIGenerator with mocked client."""
        generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")
        return generator

    def test_generate_response_passes_tools_to_api(self, ai_generator):
        """Test that tools are passed to the API when provided."""
        # Setup mock response without tool use
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [Mock(text="Direct response")]

        ai_generator.client.messages.create = Mock(return_value=mock_response)

        tools = [{"name": "search_course_content", "description": "Search courses"}]
        result = ai_generator.generate_response(
            query="What is MCP?",
            tools=tools,
            tool_manager=Mock()
        )

        # Verify API was called with tools
        call_kwargs = ai_generator.client.messages.create.call_args[1]
        assert "tools" in call_kwargs
        assert call_kwargs["tools"] == tools

    def test_generate_response_sets_tool_choice_auto(self, ai_generator):
        """Test that tool_choice is set to auto when tools are provided."""
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [Mock(text="Response")]

        ai_generator.client.messages.create = Mock(return_value=mock_response)

        tools = [{"name": "search_course_content", "description": "Search"}]
        ai_generator.generate_response(query="test", tools=tools, tool_manager=Mock())

        call_kwargs = ai_generator.client.messages.create.call_args[1]
        assert call_kwargs["tool_choice"] == {"type": "auto"}

    def test_generate_response_handles_tool_use_stop_reason(self, ai_generator):
        """Test that tool_use stop_reason triggers tool execution."""
        # Create mock tool use content block
        tool_use_block = Mock()
        tool_use_block.type = "tool_use"
        tool_use_block.name = "search_course_content"
        tool_use_block.id = "tool_123"
        tool_use_block.input = {"query": "What is MCP?"}

        # First response requests tool use
        initial_response = Mock()
        initial_response.stop_reason = "tool_use"
        initial_response.content = [tool_use_block]

        # Second response after tool execution
        final_response = Mock()
        final_response.content = [Mock(text="MCP is a protocol for...")]

        ai_generator.client.messages.create = Mock(side_effect=[initial_response, final_response])

        # Mock tool manager
        tool_manager = Mock()
        tool_manager.execute_tool = Mock(return_value="Tool result: MCP information")

        tools = [{"name": "search_course_content"}]
        result = ai_generator.generate_response(
            query="What is MCP?",
            tools=tools,
            tool_manager=tool_manager
        )

        # Verify tool was executed
        tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="What is MCP?"
        )

    def test_tool_execution_passes_correct_parameters(self, ai_generator):
        """Test that tool execution receives correct parameters from AI response."""
        tool_use_block = Mock()
        tool_use_block.type = "tool_use"
        tool_use_block.name = "search_course_content"
        tool_use_block.id = "tool_456"
        tool_use_block.input = {
            "query": "embeddings",
            "course_name": "Chroma",
            "lesson_number": 3
        }

        initial_response = Mock()
        initial_response.stop_reason = "tool_use"
        initial_response.content = [tool_use_block]

        final_response = Mock()
        final_response.content = [Mock(text="Embeddings are...")]

        ai_generator.client.messages.create = Mock(side_effect=[initial_response, final_response])

        tool_manager = Mock()
        tool_manager.execute_tool = Mock(return_value="Search results")

        ai_generator.generate_response(
            query="Tell me about embeddings in the Chroma course lesson 3",
            tools=[{"name": "search_course_content"}],
            tool_manager=tool_manager
        )

        # Verify all parameters were passed
        tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="embeddings",
            course_name="Chroma",
            lesson_number=3
        )

    def test_generate_response_without_tools_returns_direct_response(self, ai_generator):
        """Test that response without tools returns text directly."""
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [Mock(text="Direct answer without tools")]

        ai_generator.client.messages.create = Mock(return_value=mock_response)

        result = ai_generator.generate_response(query="What is 2+2?")

        assert result == "Direct answer without tools"

    def test_tool_result_sent_back_to_api(self, ai_generator):
        """Test that tool results are sent back to API for final response."""
        tool_use_block = Mock()
        tool_use_block.type = "tool_use"
        tool_use_block.name = "search_course_content"
        tool_use_block.id = "tool_789"
        tool_use_block.input = {"query": "test"}

        initial_response = Mock()
        initial_response.stop_reason = "tool_use"
        initial_response.content = [tool_use_block]

        final_response = Mock()
        final_response.content = [Mock(text="Final answer")]

        ai_generator.client.messages.create = Mock(side_effect=[initial_response, final_response])

        tool_manager = Mock()
        tool_manager.execute_tool = Mock(return_value="Tool execution result")

        ai_generator.generate_response(
            query="test",
            tools=[{"name": "search_course_content"}],
            tool_manager=tool_manager
        )

        # Check second API call includes tool results
        second_call = ai_generator.client.messages.create.call_args_list[1]
        messages = second_call[1]["messages"]

        # Should have: user message, assistant tool_use, user tool_result
        assert len(messages) == 3
        assert messages[2]["role"] == "user"
        assert messages[2]["content"][0]["type"] == "tool_result"
        assert messages[2]["content"][0]["content"] == "Tool execution result"


class TestAIGeneratorSystemPrompt:
    """Test suite for AIGenerator's system prompt configuration."""

    @pytest.fixture
    def ai_generator(self):
        """Create AIGenerator for prompt testing."""
        with patch('ai_generator.anthropic.Anthropic'):
            return AIGenerator(api_key="test-key", model="test-model")

    def test_system_prompt_mentions_search_tool(self, ai_generator):
        """Test that system prompt documents search_course_content tool."""
        assert "search_course_content" in ai_generator.SYSTEM_PROMPT

    def test_system_prompt_mentions_outline_tool(self, ai_generator):
        """Test that system prompt documents get_course_outline tool."""
        assert "get_course_outline" in ai_generator.SYSTEM_PROMPT

    def test_system_prompt_guides_tool_selection(self, ai_generator):
        """Test that system prompt provides guidance on when to use each tool."""
        prompt = ai_generator.SYSTEM_PROMPT

        # Should mention when to use outline tool
        assert "course structure" in prompt.lower() or "outline" in prompt.lower()

        # Should mention when to use search tool
        assert "content" in prompt.lower()

    def test_conversation_history_appended_to_system(self, ai_generator):
        """Test that conversation history is appended to system prompt."""
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [Mock(text="Response")]

        ai_generator.client.messages.create = Mock(return_value=mock_response)

        ai_generator.generate_response(
            query="Follow up question",
            conversation_history="User: Hi\nAssistant: Hello"
        )

        call_kwargs = ai_generator.client.messages.create.call_args[1]
        system = call_kwargs["system"]

        assert "Previous conversation:" in system
        assert "User: Hi" in system


class TestAIGeneratorToolManagerIntegration:
    """Test suite for AIGenerator integration with ToolManager."""

    @pytest.fixture
    def real_tool_manager(self, mock_vector_store, sample_search_results):
        """Create a real ToolManager with mocked vector store."""
        mock_vector_store.search.return_value = sample_search_results
        mock_vector_store.get_lesson_link.return_value = None

        tool = CourseSearchTool(mock_vector_store)
        manager = ToolManager()
        manager.register_tool(tool)
        return manager

    def test_tool_manager_executes_search_correctly(self, real_tool_manager, mock_vector_store, sample_search_results):
        """Test that ToolManager correctly executes search_course_content."""
        result = real_tool_manager.execute_tool(
            "search_course_content",
            query="What is MCP?"
        )

        # Should return formatted search results
        assert isinstance(result, str)
        mock_vector_store.search.assert_called_once()

    def test_tool_manager_returns_error_for_unknown_tool(self, real_tool_manager):
        """Test that ToolManager handles unknown tool names."""
        result = real_tool_manager.execute_tool("nonexistent_tool", query="test")

        assert "not found" in result.lower()

    def test_tool_definitions_match_expected_format(self, real_tool_manager):
        """Test that tool definitions have correct Anthropic format."""
        definitions = real_tool_manager.get_tool_definitions()

        for tool_def in definitions:
            assert "name" in tool_def
            assert "description" in tool_def
            assert "input_schema" in tool_def
            assert "type" in tool_def["input_schema"]
            assert tool_def["input_schema"]["type"] == "object"
