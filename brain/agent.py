# agent.py
import re
import json
import time
from typing import Any, Dict, Optional

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from config import OLLAMA_MODEL, OLLAMA_BASE_URL
from brain.tools import tools
from brain.session_context import set_profile
from logger import logger, log_performance, event_logger

# Map tool names → callable for the leaked-tool-call fallback
_TOOLS_MAP = {t.name: t for t in tools}


def _try_execute_leaked_tool_call(text: str, user_input: str = "") -> Optional[str]:
    """
    Some models (granite4, llama3.1, mistral) leak the tool call as either:
      - A bare tool name:  "retrieve_policy"
      - A JSON object:     {"name": "retrieve_policy", "parameters": {...}}
    Detect either form, run the tool, and return the result.
    Returns None if the text is not a leaked tool call.
    """
    stripped = text.strip()

    # 1. Bare tool name — e.g. the model just outputs "retrieve_policy"
    if stripped in _TOOLS_MAP:
        tool_name = stripped
        if tool_name == "retrieve_policy":
            args = {"query": user_input}
        elif tool_name == "recommend_courses":
            args = {"learning_goal": user_input}
        else:
            args = {}
        logger.info(f"Executing bare tool-name leak: {tool_name}({args})")
        return _TOOLS_MAP[tool_name].invoke(args)

    # 2. JSON form — {"name": ..., "parameters"/"arguments": {...}}
    if not (stripped.startswith('{') or stripped.startswith('[')):
        return None
    try:
        data = json.loads(stripped)
        if isinstance(data, list):
            data = data[0]
        tool_name = data.get("name")
        # accept both "parameters" (llama3.1) and "arguments" (mistral/openai)
        args = data.get("parameters") or data.get("arguments") or {}
        if not tool_name or tool_name not in _TOOLS_MAP:
            return None
        # Remove None values — tools have defaults for optional params
        args = {k: v for k, v in args.items() if v is not None}
        logger.info(f"Executing leaked JSON tool call: {tool_name}({args})")
        result = _TOOLS_MAP[tool_name].invoke(args)
        return result
    except Exception as e:
        logger.warning(f"Failed to parse/execute leaked tool call: {e}")
        return None


def _phrase_tool_result(llm, tool_result: str) -> str:
    """
    Clean up a raw tool result string for TTS — strip markdown symbols and
    return it directly.  We intentionally do NOT re-invoke the LLM here
    because llama3.1 ignores the instruction and generates from its own
    knowledge rather than from the tool data.
    """
    text = tool_result.strip()
    text = re.sub(r'\*{1,3}(.+?)\*{1,3}', r'\1', text)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*[-*]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'-{3,}', '', text)
    return text.strip()


def _trim_to_last_sentence(text: str) -> str:
    """
    If the text ends mid-sentence (no terminal punctuation), trim back to the
    last complete sentence so TTS never reads a dangling fragment.
    """
    # Already ends cleanly
    if re.search(r'[.!?]["\']?\s*$', text):
        return text
    # Find the last sentence-ending punctuation
    match = re.search(r'^(.*[.!?]["\']?)\s+\S', text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    # No complete sentence found — return as-is (e.g. a single short fragment)
    return text


_BASE_SYSTEM_PROMPT = """You are an HR assistant embedded in the company's Learning Management System (LMS).

You have access to the employee's LMS profile (job role, department, skill level, enrolled courses, known skills).
You do NOT need to ask the employee for this information — it is already loaded.

You have exactly three responsibilities. Each one REQUIRES a specific tool — you must NEVER answer from your own knowledge for these:

1. ANY question about HR policies, company rules, leave, benefits, or working conditions
   → ALWAYS call 'retrieve_policy'. Never answer from memory.

2. ANY request related to learning, courses, skills, career goals, or training
   → ALWAYS call 'recommend_courses'. Never suggest course names yourself.
   - learning_goal (required): what the employee wants to learn/achieve.
   - preferred_difficulty: map words like "beginner", "basic", "advanced" to
     "Beginner", "Intermediate", or "Advanced". Use the employee's skill level
     from the LMS profile if they don't state a preference.
   - preferred_duration: INFER from time mentions only.
     "Short" (<3 h/week), "Medium" (3–10 h/week), "Long" (10+ h/week).
     Omit if no time was mentioned.
   - preferred_category: extract from topic mentions ("machine learning", "python", etc.).
   - Never ask for job role, department, skills, or courses — those come from the LMS profile.

3. ANY request to be tested or assessed on a completed course
   → ALWAYS call 'generate_assessment'. Ask for the course name/ID if missing.

Always be professional and helpful.

RESPONSE FORMAT RULES — follow these strictly:
- Plain text only. No markdown. No asterisks, no bullet dashes, no bold, no headers.
- Maximum 3 sentences. If listing items, maximum 3 items, one per line, no symbols.
- Never pad with filler phrases like "I hope this helps" or "feel free to ask".
- Always base your answer on what the tool actually returned. Never ignore tool results.
- If a tool explicitly returns "No courses found" or "No relevant policy documents found", say: "I couldn't find anything on that. Please contact HR directly."
- Do not make up, guess, or hallucinate any policy details, course names, or facts not returned by a tool.

IMPORTANT: Never mention tool names, never say "I will use the X tool", never narrate what
you are about to do. Just do it silently and speak only the final answer to the employee.

If the employee's message is unclear, very short, or appears to be a transcription error
(random letters, incomplete words, or nonsense), do NOT guess or make up an answer.
Instead, respond with a single polite request for clarification, e.g.:
"Sorry, I didn't quite catch that — could you rephrase your question?"
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
        self.llm = ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.3,   # lower = faster sampling, more deterministic answers
            num_ctx=2048,      # 4096 was overkill; 2048 covers 10-turn history comfortably
            num_predict=200,   # keep responses short — long text causes lipsync to cut off
            keep_alive=0,      # unload from GPU immediately after response — frees VRAM for Wav2Lip
        )
        self._profile: Optional[Dict[str, Any]] = None
        self.messages = []
        self.llm_with_tools = self.llm.bind_tools(tools)
        logger.info("HRAgent initialized")

    def set_profile(self, profile: Dict[str, Any]) -> None:
        """Called once per session by the web layer after /session/start."""
        self._profile = profile
        set_profile(profile)   # make profile available to tools via context var
        logger.info(f"Agent profile set for user: {profile.get('user_id')}")

    @log_performance
    def run(self, user_input: str) -> str:
        """Process a single user message and return the final answer."""
        # Ensure tools can always read the profile even across turns
        if self._profile:
            set_profile(self._profile)

        _t0 = time.time()
        # Tracking fields written to the event log at the end of this method.
        _ev = {"tool_called": None, "tool_args": {}, "hallucination_guard": False}

        logger.info(f"User input: {user_input[:100]}...")
        self.messages.append(("human", user_input))
        # Keep only the last 10 turns to prevent context window growth
        trimmed = self.messages[-10:] if len(self.messages) > 10 else self.messages

        # Build the full message list: system prompt + conversation history.
        # The profile block is injected here at call-time so set_profile() no
        # longer needs to rebuild anything.
        msgs = [SystemMessage(content=_build_system_prompt(self._profile))]
        for role, content in trimmed:
            msgs.append(HumanMessage(content=content) if role == "human" else AIMessage(content=content))

        # Single LLM call — the model decides which tool to call (if any).
        # We never hand the tool result back to the LLM for a second pass; that
        # is what caused the runaway loop on granite4 (6 chained tool calls,
        # 57 s latency). We phrase the raw tool output ourselves via
        # _phrase_tool_result, which is sufficient for TTS delivery.
        response = self.llm_with_tools.invoke(msgs)
        ai_message = response.content or ""

        # Tracks whether the response is LLM-generated prose that may be cut off
        # mid-sentence.  Set to False for complete tool outputs so that
        # _trim_to_last_sentence doesn't chop off trailing list items (e.g. the
        # last set of assessment options).
        _needs_sentence_trim = True

        # 1. Model leaks the tool call — bare name ("retrieve_policy") or raw JSON.
        tool_result = _try_execute_leaked_tool_call(ai_message, user_input)
        if tool_result:
            ai_message = _phrase_tool_result(self.llm, tool_result)
            _needs_sentence_trim = False  # tool output is already complete

        # 2. Normal structured tool call — execute the FIRST one only.
        #    If the model requested several (granite4 parallel-tool behaviour),
        #    the extras are silently ignored.
        elif response.tool_calls:
            from langchain_core.messages import ToolMessage
            tool_call = response.tool_calls[0]
            clean_args = {k: v for k, v in tool_call["args"].items() if v is not None}
            _ev["tool_called"] = tool_call["name"]
            _ev["tool_args"]   = clean_args
            logger.info(f"Tool: {tool_call['name']} | args: {clean_args}")
            raw_result = _TOOLS_MAP[tool_call["name"]].invoke(clean_args)

            if tool_call["name"] == "retrieve_policy":
                # Policy docs are multi-paragraph — the raw result is too long for
                # TTS.  One extra LLM call (plain llm, no tools bound) summarises
                # it into 1-3 sentences.  Using self.llm here (not llm_with_tools)
                # means the model cannot call another tool, so this is a hard stop.
                follow_up = msgs + [
                    response,
                    ToolMessage(content=str(raw_result), tool_call_id=tool_call["id"]),
                ]
                summary = self.llm.invoke(follow_up)
                if summary.content:
                    ai_message = summary.content
                    # Summary is LLM-generated and may be cut off — trim applies.
                else:
                    ai_message = _phrase_tool_result(self.llm, raw_result)
                    _needs_sentence_trim = False  # raw tool output is complete
            else:
                # Course recommendations and assessments are already structured
                # lists — clean up and return directly without a second LLM call.
                ai_message = _phrase_tool_result(self.llm, raw_result)
                _needs_sentence_trim = False  # tool output is already complete

        # 3. Direct answer with no tool call.
        else:
            # Hallucination guard: if the model produced a course-list-shaped response
            # without calling the tool, it invented courses from its own knowledge.
            # Detect the pattern and force the correct tool call instead.
            _course_openers = ("here are some", "here's some", "recommended course",
                               "i recommend", "you might want to")
            if any(p in ai_message.lower() for p in _course_openers):
                logger.info("Hallucinated course list detected — forcing recommend_courses call")
                _ev["hallucination_guard"] = True
                _ev["tool_called"] = "recommend_courses"
                _ev["tool_args"]   = {"learning_goal": user_input}
                result = _TOOLS_MAP["recommend_courses"].invoke({"learning_goal": user_input})
                ai_message = _phrase_tool_result(self.llm, result)
                _needs_sentence_trim = False  # tool output is already complete
            else:
                # Strip any bare URLs the model may have embedded (not from tool results).
                if re.search(r'\[.+?\]\(https?://', ai_message):
                    ai_message = re.sub(r'\[([^\]]+)\]\(https?://[^)]+\)', r'\1', ai_message)

                _fallback_phrases = ("i couldn't find", "please contact hr", "no courses found")
                if not ai_message.strip() or any(p in ai_message.lower() for p in _fallback_phrases):
                    ai_message = "Sorry, I didn't quite catch that — could you rephrase your question?"

        # Mistral sometimes leaks raw tool-call tokens into the final content string —
        # either as a leading [TOOL_CALLS] prefix or a trailing JSON array of tool objects.
        ai_message = re.sub(r'^\[TOOL_CALLS\]\s*(\[.*?\])?\s*', '', ai_message, flags=re.DOTALL)
        ai_message = re.sub(r'\s*\[\s*\{\s*"name"\s*:.*?\}\s*\]\s*$', '', ai_message, flags=re.DOTALL)
        # Strip markdown formatting — TTS reads symbols aloud ("asterisk asterisk bold")
        ai_message = re.sub(r'\*{1,3}(.+?)\*{1,3}', r'\1', ai_message)   # **bold**, *italic*
        ai_message = re.sub(r'#{1,6}\s*', '', ai_message)                  # ## headers
        ai_message = re.sub(r'^\s*[-*]\s+', '', ai_message, flags=re.MULTILINE)  # - bullet points
        ai_message = re.sub(r'^\s*\d+\.\s+', '', ai_message, flags=re.MULTILINE) # 1. numbered lists
        ai_message = re.sub(r'-{3,}', '', ai_message)                      # --- dividers
        # Strip tool-narration phrases the model sometimes emits before calling a tool,
        # e.g. "To help you, I will use the 'retrieve_policy' tool." — these should never
        # reach the employee.
        ai_message = re.sub(
            r"^.*?\bI (will|am going to|'ll) use the ['\"]?\w+['\"]? tool\.?\s*",
            '',
            ai_message,
            flags=re.IGNORECASE | re.DOTALL,
        )
        ai_message = ai_message.strip()
        # If num_predict cut the response mid-sentence, trim back to the last complete sentence
        # so TTS/lipsync never speaks a dangling fragment.
        # Only applies to LLM-generated prose — tool outputs are already complete.
        if _needs_sentence_trim:
            ai_message = _trim_to_last_sentence(ai_message)
        self.messages.append(("ai", ai_message))
        logger.info(f"Agent response: {ai_message[:100]}...")

        # Structured event — one record per turn, queryable via logs/events.jsonl
        _ev.update({
            "user_id":   self._profile.get("user_id", "") if self._profile else "",
            "input":     user_input,
            "response":  ai_message,
            "latency_ms": round((time.time() - _t0) * 1000),
            "grounded":  _ev["tool_called"] is not None,
        })
        event_logger.log(_ev)
        self._last_event = _ev   # exposed for eval.py to inspect without parsing the log

        return ai_message

    def reset_conversation(self) -> None:
        """Clear conversation history (profile is kept)."""
        self.messages = []
        logger.info("Conversation reset")
