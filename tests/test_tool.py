# tests/test_tool.py
# Tests for recommend_courses and generate_assessment tools.
# Uses mocks — no real API server or Ollama needed.

from unittest.mock import patch, MagicMock


# ── helpers ──────────────────────────────────────────────────────────────────

def _mock_rag():
    """Return a mock RAGManager so tools.py doesn't hit ChromaDB on import."""
    mock = MagicMock()
    mock.retrieve.return_value = []
    return mock


_SAMPLE_PROFILE = {
    "user_id": "emp_001",
    "name": "Abiola K.",
    "job_role": "Data Analyst",
    "department": "Engineering",
    "skill_level": "Intermediate",
    "known_skills": ["SQL", "Python"],
    "enrolled_courses": ["data-101"],
    "context": "avatar_chat",
}


# ── recommend_courses ─────────────────────────────────────────────────────────

class TestRecommendCourses:

    def _call(self, mock_post, payload, status=200, profile=None):
        """Helper: patch requests.post, inject profile, and call recommend_courses."""
        mock_response = MagicMock()
        mock_response.status_code = status
        mock_response.json.return_value = payload
        mock_response.raise_for_status = MagicMock()
        if status >= 400:
            mock_response.raise_for_status.side_effect = Exception(f"HTTP {status}")
        mock_post.return_value = mock_response

        injected = profile or _SAMPLE_PROFILE

        with patch("brain.tools.rag", _mock_rag()), \
             patch("brain.tools.get_profile", return_value=injected):
            from brain.tools import recommend_courses
            return recommend_courses.invoke({
                "learning_goal": "transition to machine learning",
                "preferred_difficulty": "Beginner",
                "preferred_duration": "Short",
            })

    @patch("brain.tools.requests.post")
    def test_returns_courses_on_success(self, mock_post):
        result = self._call(mock_post, {
            "courses": [
                {"title": "ML Foundations", "description": "Intro to ML."},
                {"title": "Deep Learning with PyTorch", "description": "Hands-on DL."},
            ]
        })
        assert "ML Foundations" in result
        assert "Deep Learning with PyTorch" in result

    @patch("brain.tools.requests.post")
    def test_profile_fields_sent_to_api(self, mock_post):
        """Profile fields from LMS must appear in the API payload — not missing."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"courses": [{"title": "Test", "description": ""}]}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        with patch("brain.tools.rag", _mock_rag()), \
             patch("brain.tools.get_profile", return_value=_SAMPLE_PROFILE):
            from brain.tools import recommend_courses
            recommend_courses.invoke({"learning_goal": "learn Python"})

        call_payload = mock_post.call_args[1]["json"]
        assert call_payload["user_id"] == "emp_001"
        assert call_payload["job_role"] == "Data Analyst"
        assert call_payload["known_skills"] == ["SQL", "Python"]
        assert call_payload["learning_goal"] == "learn Python"

    @patch("brain.tools.requests.post")
    def test_missing_learning_goal_returns_message(self, mock_post):
        with patch("brain.tools.rag", _mock_rag()), \
             patch("brain.tools.get_profile", return_value=_SAMPLE_PROFILE):
            from brain.tools import recommend_courses
            result = recommend_courses.invoke({})
        assert "learning goal" in result.lower()
        mock_post.assert_not_called()

    @patch("brain.tools.requests.post")
    def test_empty_courses_list(self, mock_post):
        result = self._call(mock_post, {"courses": []})
        assert "No courses found" in result

    @patch("brain.tools.requests.post")
    def test_api_error_returns_friendly_message(self, mock_post):
        mock_post.side_effect = Exception("Connection refused")
        with patch("brain.tools.rag", _mock_rag()), \
             patch("brain.tools.get_profile", return_value=_SAMPLE_PROFILE):
            from brain.tools import recommend_courses
            result = recommend_courses.invoke({"learning_goal": "learn ML"})
        assert "service error" in result.lower() or "couldn't fetch" in result.lower()

    @patch("brain.tools.requests.post")
    def test_no_profile_still_calls_api(self, mock_post):
        """When no LMS profile is available the tool should still attempt the call."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"courses": [{"title": "Course A", "description": ""}]}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        with patch("brain.tools.rag", _mock_rag()), \
             patch("brain.tools.get_profile", return_value=None):
            from brain.tools import recommend_courses
            result = recommend_courses.invoke({"learning_goal": "learn SQL"})

        assert "Course A" in result
        call_payload = mock_post.call_args[1]["json"]
        assert call_payload["user_id"] == ""          # graceful empty default


# ── generate_assessment ───────────────────────────────────────────────────────

class TestGenerateAssessment:

    def _call(self, mock_post, payload, course_id="python-101", status=200):
        """Helper: patch requests.post and call generate_assessment tool."""
        mock_response = MagicMock()
        mock_response.status_code = status
        mock_response.json.return_value = payload
        mock_response.raise_for_status = MagicMock()
        if status >= 400:
            mock_response.raise_for_status.side_effect = Exception(f"HTTP {status}")
        mock_post.return_value = mock_response

        with patch("brain.tools.rag", _mock_rag()):
            from brain.tools import generate_assessment
            return generate_assessment.invoke({"course_id": course_id})

    @patch("brain.tools.requests.post")
    def test_returns_questions_on_success(self, mock_post):
        result = self._call(mock_post, {
            "questions": [
                {"question": "What is a list comprehension?", "options": ["A", "B", "C", "D"]},
                {"question": "What does `len()` return?"},
            ]
        })
        assert "What is a list comprehension?" in result
        assert "What does `len()` return?" in result

    @patch("brain.tools.requests.post")
    def test_options_are_included(self, mock_post):
        result = self._call(mock_post, {
            "questions": [
                {"question": "Which keyword defines a function?",
                 "options": ["def", "fun", "func", "lambda"]},
            ]
        })
        assert "def" in result
        assert "lambda" in result

    @patch("brain.tools.requests.post")
    def test_missing_course_id_returns_message(self, mock_post):
        with patch("brain.tools.rag", _mock_rag()):
            from brain.tools import generate_assessment
            result = generate_assessment.invoke({})
        assert "Course ID is required" in result
        mock_post.assert_not_called()

    @patch("brain.tools.requests.post")
    def test_empty_questions_returns_message(self, mock_post):
        result = self._call(mock_post, {"questions": []})
        assert "No assessment available" in result

    @patch("brain.tools.requests.post")
    def test_api_error_returns_friendly_message(self, mock_post):
        mock_post.side_effect = Exception("Timeout")
        with patch("brain.tools.rag", _mock_rag()):
            from brain.tools import generate_assessment
            result = generate_assessment.invoke({"course_id": "ml-basics"})
        assert "service error" in result.lower() or "couldn't generate" in result.lower()
