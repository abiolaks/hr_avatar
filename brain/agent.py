# agent.py
from langgraph.prebuilt import create_react_agent
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage
from config import OLLAMA_MODEL
from brain.tools import tools
from logger import logger, log_performance

class HRAgent:
    def __init__(self):
        self.llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.7)
        system_prompt = SystemMessage(
            content="""You are an HR assistant for a company. You have three main responsibilities:
1. Answer questions about company policies using the 'retrieve_policy' tool.
2. Help employees with career development by recommending courses. Use the 'recommend_courses' tool.
   Before calling it, you must gather the user's current role, desired role, skills they want to develop, and their time commitment.
   If any information is missing, ask the user politely.
3. Generate assessments for completed courses using the 'generate_assessment' tool.
   You need the course ID or name – ask the user if not provided.

Always be professional and helpful. If you don't know something, say so.
"""
        )
        self.agent = create_react_agent(
            model=self.llm,
            tools=tools,
            state_modifier=system_prompt
        )
        self.messages = []  # conversation history
        logger.info("HRAgent initialized")

    @log_performance
    def run(self, user_input: str) -> str:
        """Process a single user message and return the final answer."""
        logger.info(f"User input: {user_input[:100]}...")
        self.messages.append(("human", user_input))
        result = self.agent.invoke({"messages": self.messages})
        ai_message = result["messages"][-1].content
        self.messages.append(("ai", ai_message))
        logger.info(f"Agent response: {ai_message[:100]}...")
        return ai_message

    def reset_conversation(self):
        """Clear conversation history."""
        self.messages = []
        logger.info("Conversation reset")
