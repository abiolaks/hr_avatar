# tools.py
import requests
from langchain_core.tools import tool
from config import RECOMMENDATION_API_URL, ASSESSMENT_API_URL
from brain.rag import RAGManager
from brain.session_context import get_profile
from logger import logger, log_performance

# Initialize RAG manager once
rag = RAGManager()


@tool
@log_performance
def retrieve_policy(query: str) -> str:
    """
    Retrieve HR policy information from the company's internal documents.
    Use this for any question about company policies, benefits, leave, etc.
    """
    logger.info(f"Tool: retrieve_policy called with query: {query[:100]}...")
    docs = rag.retrieve(query, k=3)
    if not docs:
        return "No relevant policy documents found."
    context = "\n\n".join([doc.page_content for doc in docs])
    return f"Relevant policy information:\n{context}"


@tool
@log_performance
def recommend_courses(
    learning_goal: str = None,
    preferred_category: str = None,
    preferred_difficulty: str = None,
    preferred_duration: str = None,
) -> str:
    """
    Recommend courses based on what the employee wants to learn or achieve.

    Only call this tool once you know the employee's learning_goal.
    The other three fields (preferred_category, preferred_difficulty,
    preferred_duration) are optional — use them if the employee mentioned them.

    Do NOT ask the employee for their job role, department, skills, or enrolled
    courses — those come from their LMS profile automatically.
    """
    logger.info("Tool: recommend_courses called")

    if not learning_goal:
        return (
            "I need to know your learning goal before I can recommend courses. "
            "What would you like to learn or achieve?"
        )

    # Pull LMS profile fields injected at session start
    profile = get_profile() or {}

    payload = {
        # ── From LMS profile (no need to ask the user) ──────────────────
        "user_id": profile.get("user_id", ""),
        "name": profile.get("name", ""),
        "job_role": profile.get("job_role", ""),
        "department": profile.get("department", ""),
        "skill_level": profile.get("skill_level", "Beginner"),
        "known_skills": profile.get("known_skills", []),
        "enrolled_courses": profile.get("enrolled_courses", []),
        "context": profile.get("context", "avatar_chat"),
        # ── From the current conversation ───────────────────────────────
        "learning_goal": learning_goal,
        "preferred_category": preferred_category or "",
        "preferred_difficulty": preferred_difficulty or profile.get("skill_level", "Beginner"),
        "preferred_duration": preferred_duration or "",
    }

    logger.info(
        f"Calling recommendation API | user: {payload['user_id']} "
        f"| goal: {learning_goal}"
    )
    try:
        response = requests.post(RECOMMENDATION_API_URL, json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
        courses = data.get("courses", [])
        if not courses:
            return "No courses found matching your criteria."
        result = "Here are some recommended courses:\n"
        for i, course in enumerate(courses, 1):
            result += f"{i}. {course['title']}: {course.get('description', '')}\n"
        logger.info(f"Recommendation API returned {len(courses)} courses")
        return result
    except Exception as e:
        logger.error(f"Recommendation API error: {e}")
        return f"Sorry, I couldn't fetch recommendations due to a service error: {str(e)}"


@tool
@log_performance
def generate_assessment(course_id: str = None) -> str:
    """
    Generate a quiz or assessment for a completed course. Call this when the
    employee says they have finished a module or course and want to test their
    knowledge. You need the course_id — ask the employee if not provided.
    """
    logger.info("Tool: generate_assessment called")

    if not course_id:
        return "Course ID is required. Which course did you complete?"

    payload = {"course_id": course_id}
    logger.info(f"Calling assessment API with payload: {payload}")
    try:
        response = requests.post(ASSESSMENT_API_URL, json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
        questions = data.get("questions", [])
        if not questions:
            return "No assessment available for that course."
        result = "Here is your assessment:\n"
        for i, q in enumerate(questions, 1):
            result += f"{i}. {q['question']}\n"
            if "options" in q:
                for opt in q["options"]:
                    result += f"   - {opt}\n"
        logger.info(f"Assessment API returned {len(questions)} questions")
        return result
    except Exception as e:
        logger.error(f"Assessment API error: {e}")
        return f"Sorry, I couldn't generate an assessment due to a service error: {str(e)}"


# Export list of tools
tools = [retrieve_policy, recommend_courses, generate_assessment]
