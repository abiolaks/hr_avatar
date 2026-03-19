# agent.py
from typing import Any, Dict, Optional

from langgraph.prebuilt import create_react_agent
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage
from config import OLLAMA_MODEL
from brain.tools import tools
from brain.session_context import set_profile
from logger import logger, log_performance


_BASE_SYSTEM_PROMPT = """You are an HR assistant embedded in the company's Learning Management System (LMS).

You have access to the employee's LMS profile (job role, department, skill level, enrolled courses, known skills).
You do NOT need to ask the employee for this information — it is already loaded.

Your three responsibilities:
1. Answer questions about company policies — use the 'retrieve_policy' tool.
2. Recommend learning courses — use the 'recommend_courses' tool.
   - learning_goal: ask the employee what they want to learn or achieve if not stated.
   - preferred_difficulty: accept what the employee says ("beginner-friendly",
     "something advanced", etc.). Do NOT ask for it explicitly.
   - preferred_duration: INFER from any time mention in the conversation.
     Map to "Short" (<3 h/week), "Medium" (3–10 h/week), or "Long" (10+ h/week).
     Do NOT ask about time — if no time was mentioned, omit it.
   - preferred_category: extract from topic mentions ("data science", "cloud", etc.).
   - Never ask for job role, department, skills, or courses — those come from the LMS profile.
3. Generate assessments for completed courses — use the 'generate_assessment' tool.
   Ask for the course name or ID if not provided.

Always be professional, concise, and helpful.
"""


def _build_system_prompt(profile: Optional[Dict[str, Any]]) -> str:
    """Append a profile summary block so the LLM is aware of who it is talking to."""
    if not profile:
        return _BASE_SYSTEM_PROMPT

    known_skills = ", ".join(profile.get("known_skills", [])) or "not specified"
    enrolled = ", ".join(profile.get("enrolled_courses", [])) or "none"

    profile_block = f"""
Employee profile (from LMS — do not ask the employee for these):
- Name: {profile.get("name", "Unknown")}
- Job role: {profile.get("job_role", "Unknown")}
- Department: {profile.get("department", "Unknown")}
- Skill level: {profile.get("skill_level", "Unknown")}
- Known skills: {known_skills}
- Enrolled courses: {enrolled}
"""
    return _BASE_SYSTEM_PROMPT + profile_block


class HRAgent:
    def __init__(self):
        self.llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.7)
        self._profile: Optional[Dict[str, Any]] = None
        self.messages = []
        self._build_agent()
        logger.info("HRAgent initialized")

    def _build_agent(self):
        system_prompt = SystemMessage(content=_build_system_prompt(self._profile))
        self.agent = create_react_agent(
            model=self.llm,
            tools=tools,
            state_modifier=system_prompt,
        )

    def set_profile(self, profile: Dict[str, Any]) -> None:
        """
        Called once per session by the web layer after /session/start.
        Rebuilds the agent with a profile-aware system prompt.
        """
        self._profile = profile
        set_profile(profile)   # make profile available to tools via context var
        self._build_agent()
        logger.info(f"Agent profile set for user: {profile.get('user_id')}")

    @log_performance
    def run(self, user_input: str) -> str:
        """Process a single user message and return the final answer."""
        # Ensure tools can always read the profile even across turns
        if self._profile:
            set_profile(self._profile)

        logger.info(f"User input: {user_input[:100]}...")
        self.messages.append(("human", user_input))
        result = self.agent.invoke({"messages": self.messages})
        ai_message = result["messages"][-1].content
        self.messages.append(("ai", ai_message))
        logger.info(f"Agent response: {ai_message[:100]}...")
        return ai_message

    def reset_conversation(self) -> None:
        """Clear conversation history (profile is kept)."""
        self.messages = []
        logger.info("Conversation reset")
