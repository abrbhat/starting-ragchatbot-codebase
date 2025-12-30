"""Tests for FastAPI endpoints."""
import pytest
from unittest.mock import Mock


class TestRootEndpoint:
    """Tests for the root endpoint."""

    def test_root_returns_200(self, test_client):
        """Test that root endpoint returns 200 status."""
        response = test_client.get("/")
        assert response.status_code == 200

    def test_root_returns_json(self, test_client):
        """Test that root endpoint returns JSON response."""
        response = test_client.get("/")
        assert response.headers["content-type"] == "application/json"

    def test_root_contains_message(self, test_client):
        """Test that root endpoint contains a message."""
        response = test_client.get("/")
        data = response.json()
        assert "message" in data


class TestQueryEndpoint:
    """Tests for POST /api/query endpoint."""

    def test_query_returns_200(self, test_client):
        """Test that query endpoint returns 200 on valid request."""
        response = test_client.post(
            "/api/query",
            json={"query": "What is MCP?"}
        )
        assert response.status_code == 200

    def test_query_returns_answer(self, test_client):
        """Test that query response contains answer field."""
        response = test_client.post(
            "/api/query",
            json={"query": "What is MCP?"}
        )
        data = response.json()
        assert "answer" in data
        assert isinstance(data["answer"], str)
        assert len(data["answer"]) > 0

    def test_query_returns_sources(self, test_client):
        """Test that query response contains sources field."""
        response = test_client.post(
            "/api/query",
            json={"query": "What is MCP?"}
        )
        data = response.json()
        assert "sources" in data
        assert isinstance(data["sources"], list)

    def test_query_returns_session_id(self, test_client):
        """Test that query response contains session_id field."""
        response = test_client.post(
            "/api/query",
            json={"query": "What is MCP?"}
        )
        data = response.json()
        assert "session_id" in data
        assert isinstance(data["session_id"], str)

    def test_query_creates_session_when_not_provided(self, test_client, mock_rag_system):
        """Test that a new session is created when not provided."""
        response = test_client.post(
            "/api/query",
            json={"query": "Test query"}
        )
        data = response.json()
        assert data["session_id"] == "test-session-123"
        mock_rag_system.session_manager.create_session.assert_called_once()

    def test_query_uses_provided_session_id(self, test_client, mock_rag_system):
        """Test that provided session_id is used."""
        response = test_client.post(
            "/api/query",
            json={"query": "Test query", "session_id": "existing-session"}
        )
        data = response.json()
        assert data["session_id"] == "existing-session"
        mock_rag_system.query.assert_called_with("Test query", "existing-session")

    def test_query_passes_query_to_rag_system(self, test_client, mock_rag_system):
        """Test that query is passed to RAG system."""
        test_client.post(
            "/api/query",
            json={"query": "What is the Model Context Protocol?"}
        )
        call_args = mock_rag_system.query.call_args[0]
        assert call_args[0] == "What is the Model Context Protocol?"

    def test_query_missing_query_field_returns_422(self, test_client):
        """Test that missing query field returns 422 validation error."""
        response = test_client.post(
            "/api/query",
            json={}
        )
        assert response.status_code == 422

    def test_query_invalid_json_returns_422(self, test_client):
        """Test that invalid JSON returns 422."""
        response = test_client.post(
            "/api/query",
            content="not valid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422

    def test_query_empty_query_accepted(self, test_client):
        """Test that empty query string is accepted (validation at app level)."""
        response = test_client.post(
            "/api/query",
            json={"query": ""}
        )
        assert response.status_code == 200


class TestQueryEndpointErrors:
    """Tests for error handling in query endpoint."""

    def test_query_rag_system_error_returns_500(self, test_app, mock_rag_system):
        """Test that RAG system errors return 500."""
        from fastapi.testclient import TestClient

        mock_rag_system.query.side_effect = Exception("Database connection failed")
        client = TestClient(test_app)

        response = client.post(
            "/api/query",
            json={"query": "Test query"}
        )
        assert response.status_code == 500
        assert "Database connection failed" in response.json()["detail"]


class TestCoursesEndpoint:
    """Tests for GET /api/courses endpoint."""

    def test_courses_returns_200(self, test_client):
        """Test that courses endpoint returns 200."""
        response = test_client.get("/api/courses")
        assert response.status_code == 200

    def test_courses_returns_total_courses(self, test_client):
        """Test that courses response contains total_courses."""
        response = test_client.get("/api/courses")
        data = response.json()
        assert "total_courses" in data
        assert isinstance(data["total_courses"], int)

    def test_courses_returns_course_titles(self, test_client):
        """Test that courses response contains course_titles."""
        response = test_client.get("/api/courses")
        data = response.json()
        assert "course_titles" in data
        assert isinstance(data["course_titles"], list)

    def test_courses_returns_expected_data(self, test_client):
        """Test that courses returns expected mock data."""
        response = test_client.get("/api/courses")
        data = response.json()
        assert data["total_courses"] == 2
        assert "Course A" in data["course_titles"]
        assert "Course B" in data["course_titles"]

    def test_courses_calls_get_course_analytics(self, test_client, mock_rag_system):
        """Test that courses endpoint calls get_course_analytics."""
        test_client.get("/api/courses")
        mock_rag_system.get_course_analytics.assert_called_once()


class TestCoursesEndpointErrors:
    """Tests for error handling in courses endpoint."""

    def test_courses_error_returns_500(self, test_app, mock_rag_system):
        """Test that analytics errors return 500."""
        from fastapi.testclient import TestClient

        mock_rag_system.get_course_analytics.side_effect = Exception("Analytics failed")
        client = TestClient(test_app)

        response = client.get("/api/courses")
        assert response.status_code == 500
        assert "Analytics failed" in response.json()["detail"]


class TestAPIContentTypes:
    """Tests for API content type handling."""

    def test_query_accepts_json(self, test_client):
        """Test that query endpoint accepts JSON content type."""
        response = test_client.post(
            "/api/query",
            json={"query": "Test"},
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 200

    def test_query_response_is_json(self, test_client):
        """Test that query response is JSON."""
        response = test_client.post(
            "/api/query",
            json={"query": "Test"}
        )
        assert "application/json" in response.headers["content-type"]

    def test_courses_response_is_json(self, test_client):
        """Test that courses response is JSON."""
        response = test_client.get("/api/courses")
        assert "application/json" in response.headers["content-type"]


class TestAPISessionManagement:
    """Tests for session management through API."""

    def test_session_continuity(self, test_client, mock_rag_system):
        """Test that session ID can be reused across requests."""
        # First request creates session
        response1 = test_client.post(
            "/api/query",
            json={"query": "First question"}
        )
        session_id = response1.json()["session_id"]

        # Second request uses same session
        response2 = test_client.post(
            "/api/query",
            json={"query": "Follow up", "session_id": session_id}
        )

        assert response2.json()["session_id"] == session_id

    def test_different_sessions_isolated(self, test_client, mock_rag_system):
        """Test that different session IDs are handled separately."""
        test_client.post(
            "/api/query",
            json={"query": "Question 1", "session_id": "session-a"}
        )
        test_client.post(
            "/api/query",
            json={"query": "Question 2", "session_id": "session-b"}
        )

        calls = mock_rag_system.query.call_args_list
        assert calls[0][0][1] == "session-a"
        assert calls[1][0][1] == "session-b"
