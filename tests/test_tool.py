# tests/test_tool.py
# Tests for recommend_courses and generate_assessment tools
# Uses mocks — no real API server needed

from unittest.mock import patch, MagicMock


# ── helpers ──────────────────────────────────────────────────────────────────

def _mock_rag():
    """Return a mock RAGManager so tools.py doesn't hit ChromaDB on import."""
    mock = MagicMock()
    mock.retrieve.return_value = []
    return mock


# ── recommend_courses ─────────────────────────────────────────────────────────

class TestRecommendCourses:

    def _call(self, mock_post, payload, status=200):
        """Helper: patch requests.post and call recommend_courses tool."""
        mock_response = MagicMock()
        mock_response.status_code = status
        mock_response.json.return_value = payload
        mock_response.raise_for_status = MagicMock()
        if status >= 400:
            mock_response.raise_for_status.side_effect = Exception(f"HTTP {status}")
        mock_post.return_value = mock_response

        with patch("brain.tools.rag", _mock_rag()):
            from brain.tools import recommend_courses
            return recommend_courses.invoke({
                "current_role": "Data Analyst",
                "desired_role": "ML Engineer",
                "skills_to_develop": "Python, PyTorch",
                "time_commitment": "5 hours/week",
            })

    @patch("brain.tools.requests.post")
    def test_returns_courses_on_success(self, mock_post):
        result = self._call(mock_post, {
            "courses": [
                {"title": "Deep Learning with PyTorch", "description": "Hands-on DL course."},
                {"title": "MLOps Fundamentals", "description": "Deploy ML models."},
            ]
        })
        assert "Deep Learning with PyTorch" in result
        assert "MLOps Fundamentals" in result

    @patch("brain.tools.requests.post")
    def test_missing_parameters_returns_message(self, mock_post):
        with patch("brain.tools.rag", _mock_rag()):
            from brain.tools import recommend_courses
            result = recommend_courses.invoke({
                "current_role": "Data Analyst",
                # missing desired_role, skills_to_develop, time_commitment
            })
        assert "Missing required parameters" in result
        mock_post.assert_not_called()

    @patch("brain.tools.requests.post")
    def test_empty_courses_list(self, mock_post):
        result = self._call(mock_post, {"courses": []})
        assert "No courses found" in result

    @patch("brain.tools.requests.post")
    def test_api_error_returns_friendly_message(self, mock_post):
        mock_post.side_effect = Exception("Connection refused")
        with patch("brain.tools.rag", _mock_rag()):
            from brain.tools import recommend_courses
            result = recommend_courses.invoke({
                "current_role": "Analyst",
                "desired_role": "Engineer",
                "skills_to_develop": "Python",
                "time_commitment": "3 hours/week",
            })
        assert "service error" in result.lower() or "couldn't fetch" in result.lower()


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
