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
        with patch("ai_generator.anthropic.Anthropic") as mock:
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
            query="What is MCP?", tools=tools, tool_manager=Mock()
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

        ai_generator.client.messages.create = Mock(
            side_effect=[initial_response, final_response]
        )

        # Mock tool manager
        tool_manager = Mock()
        tool_manager.execute_tool = Mock(return_value="Tool result: MCP information")

        tools = [{"name": "search_course_content"}]
        result = ai_generator.generate_response(
            query="What is MCP?", tools=tools, tool_manager=tool_manager
        )

        # Verify tool was executed
        tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", query="What is MCP?"
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
            "lesson_number": 3,
        }

        initial_response = Mock()
        initial_response.stop_reason = "tool_use"
        initial_response.content = [tool_use_block]

        final_response = Mock()
        final_response.content = [Mock(text="Embeddings are...")]

        ai_generator.client.messages.create = Mock(
            side_effect=[initial_response, final_response]
        )

        tool_manager = Mock()
        tool_manager.execute_tool = Mock(return_value="Search results")

        ai_generator.generate_response(
            query="Tell me about embeddings in the Chroma course lesson 3",
            tools=[{"name": "search_course_content"}],
            tool_manager=tool_manager,
        )

        # Verify all parameters were passed
        tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="embeddings",
            course_name="Chroma",
            lesson_number=3,
        )

    def test_generate_response_without_tools_returns_direct_response(
        self, ai_generator
    ):
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

        ai_generator.client.messages.create = Mock(
            side_effect=[initial_response, final_response]
        )

        tool_manager = Mock()
        tool_manager.execute_tool = Mock(return_value="Tool execution result")

        ai_generator.generate_response(
            query="test",
            tools=[{"name": "search_course_content"}],
            tool_manager=tool_manager,
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
        with patch("ai_generator.anthropic.Anthropic"):
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
            conversation_history="User: Hi\nAssistant: Hello",
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

    def test_tool_manager_executes_search_correctly(
        self, real_tool_manager, mock_vector_store, sample_search_results
    ):
        """Test that ToolManager correctly executes search_course_content."""
        result = real_tool_manager.execute_tool(
            "search_course_content", query="What is MCP?"
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


class TestAIGeneratorSequentialToolCalling:
    """Test suite for AIGenerator's sequential tool calling mechanism."""

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

    def _create_tool_use_response(self, tool_name: str, tool_id: str, tool_input: dict, stop_reason: str = "tool_use"):
        """Helper to create a mock tool use response."""
        tool_use_block = Mock()
        tool_use_block.type = "tool_use"
        tool_use_block.name = tool_name
        tool_use_block.id = tool_id
        tool_use_block.input = tool_input

        response = Mock()
        response.stop_reason = stop_reason
        response.content = [tool_use_block]
        return response

    def _create_text_response(self, text: str, stop_reason: str = "end_turn"):
        """Helper to create a mock text response."""
        text_block = Mock()
        text_block.text = text
        text_block.type = "text"

        response = Mock()
        response.stop_reason = stop_reason
        response.content = [text_block]
        return response

    def test_two_sequential_tool_calls_success(self, ai_generator):
        """Test that two sequential tool calls execute correctly with 3 API calls."""
        # First response: get_course_outline tool use
        first_tool_response = self._create_tool_use_response(
            "get_course_outline", "tool_1", {"course_title": "MCP"}
        )

        # Second response: search_course_content tool use
        second_tool_response = self._create_tool_use_response(
            "search_course_content", "tool_2", {"query": "MCP basics"}
        )

        # Third response: final text answer
        final_response = self._create_text_response("MCP is a protocol that...")

        ai_generator.client.messages.create = Mock(
            side_effect=[first_tool_response, second_tool_response, final_response]
        )

        tool_manager = Mock()
        tool_manager.execute_tool = Mock(side_effect=[
            "Course: MCP, Lessons: 1, 2, 3",
            "MCP stands for Model Context Protocol"
        ])

        tools = [{"name": "get_course_outline"}, {"name": "search_course_content"}]
        result = ai_generator.generate_response(
            query="What is MCP?",
            tools=tools,
            tool_manager=tool_manager
        )

        # Should make 3 API calls
        assert ai_generator.client.messages.create.call_count == 3

        # Both tools should be executed
        assert tool_manager.execute_tool.call_count == 2

        # Final result should be the text response
        assert result == "MCP is a protocol that..."

    def test_early_termination_no_second_tool_use(self, ai_generator):
        """Test that loop exits early when response doesn't request tool use."""
        # First response requests tool use
        first_response = self._create_tool_use_response(
            "search_course_content", "tool_1", {"query": "test"}
        )

        # Second response returns text (no tool use)
        second_response = self._create_text_response("Direct answer here")

        ai_generator.client.messages.create = Mock(
            side_effect=[first_response, second_response]
        )

        tool_manager = Mock()
        tool_manager.execute_tool = Mock(return_value="Search results")

        result = ai_generator.generate_response(
            query="test",
            tools=[{"name": "search_course_content"}],
            tool_manager=tool_manager
        )

        # Should only make 2 API calls (exits early)
        assert ai_generator.client.messages.create.call_count == 2

        # Only one tool executed
        assert tool_manager.execute_tool.call_count == 1

        assert result == "Direct answer here"

    def test_max_rounds_limit_enforced(self, ai_generator):
        """Test that tool calling stops at MAX_TOOL_ROUNDS and final call has no tools."""
        # Both responses request tool use (to test limit enforcement)
        first_response = self._create_tool_use_response(
            "get_course_outline", "tool_1", {"course_title": "MCP"}
        )
        second_response = self._create_tool_use_response(
            "search_course_content", "tool_2", {"query": "MCP"}
        )
        final_response = self._create_text_response("Final answer after max rounds")

        ai_generator.client.messages.create = Mock(
            side_effect=[first_response, second_response, final_response]
        )

        tool_manager = Mock()
        tool_manager.execute_tool = Mock(return_value="Tool result")

        result = ai_generator.generate_response(
            query="test",
            tools=[{"name": "test_tool"}],
            tool_manager=tool_manager
        )

        # Should make exactly 3 API calls (2 with tools + 1 final without tools)
        assert ai_generator.client.messages.create.call_count == 3

        # Final call should NOT have tools
        final_call_kwargs = ai_generator.client.messages.create.call_args_list[2][1]
        assert "tools" not in final_call_kwargs

        assert result == "Final answer after max rounds"

    def test_tool_execution_error_handling(self, ai_generator):
        """Test that tool execution errors are caught and passed to Claude."""
        tool_response = self._create_tool_use_response(
            "search_course_content", "tool_1", {"query": "test"}
        )
        final_response = self._create_text_response("I encountered an error...")

        ai_generator.client.messages.create = Mock(
            side_effect=[tool_response, final_response]
        )

        tool_manager = Mock()
        tool_manager.execute_tool = Mock(side_effect=Exception("Database connection failed"))

        result = ai_generator.generate_response(
            query="test",
            tools=[{"name": "search_course_content"}],
            tool_manager=tool_manager
        )

        # Should still make follow-up call with error message
        assert ai_generator.client.messages.create.call_count == 2

        # Check that error was passed in tool result
        second_call_messages = ai_generator.client.messages.create.call_args_list[1][1]["messages"]
        tool_result_message = second_call_messages[-1]  # Last message should be tool result
        assert "Error executing tool" in tool_result_message["content"][0]["content"]

    def test_conversation_context_preserved(self, ai_generator):
        """Test that messages accumulate correctly across tool rounds."""
        first_response = self._create_tool_use_response(
            "get_course_outline", "tool_1", {"course_title": "MCP"}
        )
        second_response = self._create_tool_use_response(
            "search_course_content", "tool_2", {"query": "MCP details"}
        )
        final_response = self._create_text_response("Complete answer")

        ai_generator.client.messages.create = Mock(
            side_effect=[first_response, second_response, final_response]
        )

        tool_manager = Mock()
        tool_manager.execute_tool = Mock(side_effect=["Outline result", "Search result"])

        ai_generator.generate_response(
            query="What is MCP?",
            tools=[{"name": "test"}],
            tool_manager=tool_manager
        )

        # Check final call has correct message accumulation
        # Should have: user query, assistant tool_use 1, user tool_result 1,
        #              assistant tool_use 2, user tool_result 2
        final_call_messages = ai_generator.client.messages.create.call_args_list[2][1]["messages"]
        assert len(final_call_messages) == 5

        # Verify message structure
        assert final_call_messages[0]["role"] == "user"  # Original query
        assert final_call_messages[1]["role"] == "assistant"  # First tool use
        assert final_call_messages[2]["role"] == "user"  # First tool result
        assert final_call_messages[3]["role"] == "assistant"  # Second tool use
        assert final_call_messages[4]["role"] == "user"  # Second tool result

    def test_no_tool_manager_returns_direct_response(self, ai_generator):
        """Test that without tool_manager, tool_use responses return directly."""
        # Even with tool_use stop_reason, no execution without tool_manager
        tool_response = self._create_tool_use_response(
            "search_course_content", "tool_1", {"query": "test"}
        )
        # Add a text block to the response for fallback
        text_block = Mock()
        text_block.text = "Would search for..."
        text_block.type = "text"
        tool_response.content.insert(0, text_block)

        ai_generator.client.messages.create = Mock(return_value=tool_response)

        # No tool_manager provided
        result = ai_generator.generate_response(
            query="test",
            tools=[{"name": "search_course_content"}],
            tool_manager=None
        )

        # Should return without executing tools
        assert ai_generator.client.messages.create.call_count == 1
        assert result == "Would search for..."

    def test_system_prompt_allows_sequential_tool_calling(self, ai_generator):
        """Test that system prompt no longer restricts to one tool per query."""
        prompt = ai_generator.SYSTEM_PROMPT

        # Should NOT have the old restriction
        assert "One tool use per query maximum" not in prompt

        # Should have new sequential guidance
        assert "sequentially" in prompt.lower() or "sequential" in prompt.lower()
