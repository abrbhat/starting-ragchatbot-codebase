"""Tests for CourseSearchTool.execute() method outputs."""

import pytest
from unittest.mock import Mock
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


class TestCourseSearchToolExecute:
    """Test suite for CourseSearchTool.execute() method."""

    def test_execute_returns_formatted_results_on_success(
        self, course_search_tool, mock_vector_store, sample_search_results
    ):
        """Test that execute returns properly formatted results when search succeeds."""
        mock_vector_store.search.return_value = sample_search_results
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson1"

        result = course_search_tool.execute(query="What is MCP?")

        # Verify search was called with correct parameters
        mock_vector_store.search.assert_called_once_with(
            query="What is MCP?", course_name=None, lesson_number=None
        )

        # Verify result contains course title and content
        assert "MCP: Build Rich-Context AI Apps" in result
        assert "MCP allows Claude to connect" in result
        assert "Lesson 1" in result

    def test_execute_with_course_name_filter(
        self, course_search_tool, mock_vector_store, sample_search_results
    ):
        """Test execute with course_name filter parameter."""
        mock_vector_store.search.return_value = sample_search_results
        mock_vector_store.get_lesson_link.return_value = None

        result = course_search_tool.execute(query="architecture", course_name="MCP")

        # Verify search was called with course_name
        mock_vector_store.search.assert_called_once_with(
            query="architecture", course_name="MCP", lesson_number=None
        )
        assert isinstance(result, str)

    def test_execute_with_lesson_number_filter(
        self, course_search_tool, mock_vector_store, sample_search_results
    ):
        """Test execute with lesson_number filter parameter."""
        mock_vector_store.search.return_value = sample_search_results
        mock_vector_store.get_lesson_link.return_value = None

        result = course_search_tool.execute(query="introduction", lesson_number=1)

        # Verify search was called with lesson_number
        mock_vector_store.search.assert_called_once_with(
            query="introduction", course_name=None, lesson_number=1
        )

    def test_execute_with_both_filters(
        self, course_search_tool, mock_vector_store, sample_search_results
    ):
        """Test execute with both course_name and lesson_number filters."""
        mock_vector_store.search.return_value = sample_search_results
        mock_vector_store.get_lesson_link.return_value = None

        result = course_search_tool.execute(
            query="tools", course_name="MCP", lesson_number=3
        )

        mock_vector_store.search.assert_called_once_with(
            query="tools", course_name="MCP", lesson_number=3
        )

    def test_execute_returns_error_message_on_search_error(
        self, course_search_tool, mock_vector_store, error_search_results
    ):
        """Test that execute returns error message when search fails."""
        mock_vector_store.search.return_value = error_search_results

        result = course_search_tool.execute(
            query="anything", course_name="NonExistentCourse"
        )

        assert "No course found matching 'NonExistentCourse'" in result

    def test_execute_returns_no_content_message_on_empty_results(
        self, course_search_tool, mock_vector_store, empty_search_results
    ):
        """Test that execute returns appropriate message when no content found."""
        mock_vector_store.search.return_value = empty_search_results

        result = course_search_tool.execute(query="nonexistent topic")

        assert "No relevant content found" in result

    def test_execute_empty_results_with_course_filter_shows_course_name(
        self, course_search_tool, mock_vector_store, empty_search_results
    ):
        """Test that empty results message includes course name when filtered."""
        mock_vector_store.search.return_value = empty_search_results

        result = course_search_tool.execute(query="xyz", course_name="MCP")

        assert "No relevant content found" in result
        assert "in course 'MCP'" in result

    def test_execute_empty_results_with_lesson_filter_shows_lesson_number(
        self, course_search_tool, mock_vector_store, empty_search_results
    ):
        """Test that empty results message includes lesson number when filtered."""
        mock_vector_store.search.return_value = empty_search_results

        result = course_search_tool.execute(query="xyz", lesson_number=5)

        assert "No relevant content found" in result
        assert "in lesson 5" in result

    def test_execute_populates_last_sources(
        self, course_search_tool, mock_vector_store, sample_search_results
    ):
        """Test that execute populates last_sources for UI display."""
        mock_vector_store.search.return_value = sample_search_results
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson"

        # Clear any previous sources
        course_search_tool.last_sources = []

        result = course_search_tool.execute(query="What is MCP?")

        # Verify sources were populated
        assert len(course_search_tool.last_sources) > 0
        assert "MCP: Build Rich-Context AI Apps" in course_search_tool.last_sources[0]

    def test_execute_sources_include_lesson_url_when_available(
        self, course_search_tool, mock_vector_store, sample_search_results
    ):
        """Test that sources include lesson URL when available."""
        mock_vector_store.search.return_value = sample_search_results
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson1"

        course_search_tool.last_sources = []
        result = course_search_tool.execute(query="What is MCP?")

        # Check that at least one source contains URL
        has_url = any(
            "https://" in source for source in course_search_tool.last_sources
        )
        assert has_url, "Sources should include lesson URLs when available"


class TestCourseSearchToolDefinition:
    """Test suite for CourseSearchTool tool definition."""

    def test_get_tool_definition_returns_valid_schema(self, course_search_tool):
        """Test that tool definition has required Anthropic schema fields."""
        definition = course_search_tool.get_tool_definition()

        assert "name" in definition
        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition

    def test_tool_definition_has_required_query_parameter(self, course_search_tool):
        """Test that query is marked as required in the schema."""
        definition = course_search_tool.get_tool_definition()
        schema = definition["input_schema"]

        assert "required" in schema
        assert "query" in schema["required"]

    def test_tool_definition_has_optional_parameters(self, course_search_tool):
        """Test that course_name and lesson_number are optional parameters."""
        definition = course_search_tool.get_tool_definition()
        properties = definition["input_schema"]["properties"]
        required = definition["input_schema"]["required"]

        assert "course_name" in properties
        assert "lesson_number" in properties
        assert "course_name" not in required
        assert "lesson_number" not in required


class TestCourseSearchToolFormatResults:
    """Test suite for result formatting."""

    def test_format_results_includes_course_header(
        self, course_search_tool, sample_search_results
    ):
        """Test that formatted results include course title in header."""
        course_search_tool.store = Mock()
        course_search_tool.store.get_lesson_link.return_value = None

        result = course_search_tool._format_results(sample_search_results)

        assert "[MCP: Build Rich-Context AI Apps" in result

    def test_format_results_includes_lesson_number_in_header(
        self, course_search_tool, sample_search_results
    ):
        """Test that formatted results include lesson number in header."""
        course_search_tool.store = Mock()
        course_search_tool.store.get_lesson_link.return_value = None

        result = course_search_tool._format_results(sample_search_results)

        assert "Lesson 1" in result or "Lesson 2" in result

    def test_format_results_separates_multiple_results(
        self, course_search_tool, sample_search_results
    ):
        """Test that multiple results are properly separated."""
        course_search_tool.store = Mock()
        course_search_tool.store.get_lesson_link.return_value = None

        result = course_search_tool._format_results(sample_search_results)

        # Should have two results separated by double newlines
        assert "\n\n" in result

    def test_format_results_handles_missing_lesson_number(self, course_search_tool):
        """Test formatting when lesson_number is None in metadata."""
        course_search_tool.store = Mock()
        course_search_tool.store.get_lesson_link.return_value = None

        results = SearchResults(
            documents=["Some content without lesson info"],
            metadata=[{"course_title": "Test Course", "lesson_number": None}],
            distances=[0.1],
        )

        result = course_search_tool._format_results(results)

        assert "[Test Course]" in result
        assert "Lesson" not in result  # Should not include "Lesson None"
