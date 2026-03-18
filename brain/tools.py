# tools.py
import requests
from langchain_core.tools import tool
from config import RECOMMENDATION_API_URL, ASSESSMENT_API_URL
from brain.rag import RAGManager
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
    current_role: str = None,
    desired_role: str = None,
    skills_to_develop: str = None,
    time_commitment: str = None,
) -> str:
    """
    Recommend courses based on career goals. Call this when the user wants course
    recommendations for career development. You must gather current_role, desired_role,
    skills_to_develop, and time_commitment before calling. If any are missing, ask the user.
    """
    logger.info("Tool: recommend_courses called")
    # Validate required parameters
    required = {"current_role", "desired_role", "skills_to_develop", "time_commitment"}
    provided = {k: v for k, v in locals().items() if k in required and v is not None}
    missing = required - set(provided.keys())
    if missing:
        msg = f"Missing required parameters: {', '.join(missing)}. Please ask the user for these."
        logger.warning(msg)
        return msg

    payload = {
        "current_role": current_role,
        "desired_role": desired_role,
        "skills": skills_to_develop,
        "time_commitment": time_commitment,
    }
    logger.info(f"Calling recommendation API with payload: {payload}")
    try:
        response = requests.post(RECOMMENDATION_API_URL, json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
        courses = data.get("courses", [])
        if not courses:
            return "No courses found matching your criteria."
        result = "Here are some recommended courses:\n"
        for i, course in enumerate(courses, 1):
            result += f"{i}. {course['title']}: {course['description']}\n"
        logger.info(f"Recommendation API returned {len(courses)} courses")
        return result
    except requests.exceptions.RequestException as e:
        logger.error(f"Recommendation API error: {e}")
        return (
            f"Sorry, I couldn't fetch recommendations due to a service error: {str(e)}"
        )


@tool
@log_performance
def generate_assessment(course_id: str = None) -> str:
    """
    Generate a quiz or assessment for a completed course. Call this when the user
    says they have finished a module or course and want to test their knowledge.
    You need the course_id (or course name) – ask the user if not provided.
    """
    logger.info("Tool: generate_assessment called")
    if not course_id:
        msg = "Course ID is required. Please ask the user which course they completed."
        logger.warning(msg)
        return msg

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
    except requests.exceptions.RequestException as e:
        logger.error(f"Assessment API error: {e}")
        return (
            f"Sorry, I couldn't generate an assessment due to a service error: {str(e)}"
        )


# Export list of tools
tools = [retrieve_policy, recommend_courses, generate_assessment]
