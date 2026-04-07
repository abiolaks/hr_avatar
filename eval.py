#!/usr/bin/env python3
"""
eval.py — End-to-end evaluation harness for the HR Avatar agent.

Runs a fixed set of gold-standard test cases against the live system
(Ollama + mock_services must both be running) and reports:

  • Tool routing accuracy  — was the correct tool called for each input?
  • Response quality       — does the response pass the content check?
  • Hallucination guard    — how many times did the safety net fire?
  • Latency                — per-case, average, and p95

Usage:
    python eval.py

Prerequisites:
    python mock_services.py   (port 8001, separate terminal)
    ollama serve              (port 11434, Ollama must be running)
"""

import sys
import time
import statistics
import requests
from datetime import datetime

sys.path.insert(0, ".")


# ── Pre-flight checks ─────────────────────────────────────────────────────────

def _check_services():
    errors = []
    try:
        requests.get("http://localhost:11434/api/tags", timeout=3)
    except Exception:
        errors.append("Ollama is not reachable at http://localhost:11434 — run: ollama serve")
    try:
        requests.get("http://localhost:8001/health", timeout=3)
    except Exception:
        errors.append("Mock services not reachable at http://localhost:8004 — run: python mock_services.py")
    if errors:
        print("\n[EVAL] Prerequisites not met:\n")
        for e in errors:
            print(f"  ✗  {e}")
        print()
        sys.exit(1)


# ── Test cases ────────────────────────────────────────────────────────────────
# Each case:
#   name          — display label
#   input         — employee message
#   expected_tool — tool that must be called (None = no tool expected)
#   quality_check — callable(response: str) -> bool
#                   True = response is correct / grounded in real data

def _has_url(r):
    """Response contains at least one real course URL from the catalog."""
    return "](https://" in r or "](http://" in r

def _mentions(words):
    return lambda r: any(w in r.lower() for w in words)

TEST_CASES = [
    # ── Policy queries ────────────────────────────────────────────────────────
    {
        "name":          "Policy: annual leave entitlement",
        "input":         "How many days of annual leave do I get?",
        "expected_tool": "retrieve_policy",
        "quality_check": _mentions(["20", "annual", "leave", "days"]),
    },
    {
        "name":          "Policy: sick leave",
        "input":         "What happens if I'm ill and can't come to work?",
        "expected_tool": "retrieve_policy",
        "quality_check": _mentions(["sick", "leave", "days", "ill"]),
    },
    {
        "name":          "Policy: working hours",
        "input":         "What are the standard working hours?",
        "expected_tool": "retrieve_policy",
        "quality_check": _mentions(["hour", "monday", "friday", "9", "5", "40"]),
    },
    {
        "name":          "Policy: maternity leave",
        "input":         "What is the maternity leave policy?",
        "expected_tool": "retrieve_policy",
        "quality_check": _mentions(["maternity", "leave", "eligible"]),
    },

    # ── Course recommendations ────────────────────────────────────────────────
    {
        "name":          "Courses: ML engineer goal",
        "input":         "My goal is to become a machine learning engineer.",
        "expected_tool": "recommend_courses",
        "quality_check": _has_url,
    },
    {
        "name":          "Courses: Python from scratch",
        "input":         "I want to learn Python. I've never coded before.",
        "expected_tool": "recommend_courses",
        "quality_check": _has_url,
    },
    {
        "name":          "Courses: AI agents beginner",
        "input":         "I want to learn about AI agents. I'm a complete beginner.",
        "expected_tool": "recommend_courses",
        "quality_check": _has_url,
    },
    {
        "name":          "Courses: advanced Python, limited time",
        "input":         "I want to master advanced Python. I only have a few hours a week.",
        "expected_tool": "recommend_courses",
        "quality_check": _has_url,
    },
    {
        "name":          "Courses: data science career",
        "input":         "I want to transition into data science.",
        "expected_tool": "recommend_courses",
        "quality_check": _has_url,
    },
    {
        "name":          "Courses: deep learning advanced",
        "input":         "I want to go deep into neural networks at an advanced level.",
        "expected_tool": "recommend_courses",
        "quality_check": _has_url,
    },

    # ── Assessments ───────────────────────────────────────────────────────────
    {
        "name":          "Assessment: course ID provided",
        "input":         "I finished course ML-101. I'd like to take the assessment.",
        "expected_tool": "generate_assessment",
        "quality_check": _mentions(["assessment", "question", "concept", "confident"]),
    },
    {
        "name":          "Assessment: no course ID",
        "input":         "I want to be tested on what I've learned.",
        "expected_tool": "generate_assessment",
        "quality_check": _mentions(["course", "which", "id", "assessment", "name"]),
    },

    # ── Garbled / unclear input ───────────────────────────────────────────────
    {
        "name":          "Garbled: keyboard mash",
        "input":         "asdfjkl qwerty zxcvbn",
        "expected_tool": None,
        "quality_check": _mentions(["rephrase", "catch", "unclear", "repeat"]),
    },
    {
        "name":          "Garbled: single letter",
        "input":         "x",
        "expected_tool": None,
        "quality_check": _mentions(["rephrase", "catch", "unclear", "repeat", "question"]),
    },

    # ── Edge cases ────────────────────────────────────────────────────────────
    {
        "name":          "Edge: difficulty + category stated",
        "input":         "I'm an advanced learner. I want to study reinforcement learning.",
        "expected_tool": "recommend_courses",
        "quality_check": _has_url,
    },
]

# ── Profile used for all test cases ──────────────────────────────────────────

_PROFILE = {
    "user_id":         "eval_user",
    "name":            "Eval User",
    "job_role":        "Software Engineer",
    "department":      "Engineering",
    "skill_level":     "Intermediate",
    "known_skills":    ["Python", "SQL"],
    "enrolled_courses": [],
    "context":         "avatar_chat",
}


# ── Runner ────────────────────────────────────────────────────────────────────

def _run_case(case: dict) -> dict:
    """Spin up a fresh agent (clean history) and run a single test case."""
    from brain.agent import HRAgent
    agent = HRAgent()
    agent.set_profile(_PROFILE)

    t0 = time.time()
    response = agent.run(case["input"])
    latency  = round(time.time() - t0, 2)

    ev = getattr(agent, "_last_event", {})
    tool_called = ev.get("tool_called")

    route_ok   = tool_called == case["expected_tool"]
    quality_ok = case["quality_check"](response)

    return {
        "name":              case["name"],
        "input":             case["input"],
        "expected_tool":     case["expected_tool"],
        "tool_called":       tool_called,
        "route_ok":          route_ok,
        "quality_ok":        quality_ok,
        "hallucination_guard": ev.get("hallucination_guard", False),
        "latency":           latency,
        "response":          response,
    }


# ── Reporting ─────────────────────────────────────────────────────────────────

_GREEN  = "\033[32m"
_RED    = "\033[31m"
_YELLOW = "\033[33m"
_RESET  = "\033[0m"
_BOLD   = "\033[1m"

def _tick(ok: bool) -> str:
    return f"{_GREEN}✓{_RESET}" if ok else f"{_RED}✗{_RESET}"

def _fmt_tool(name) -> str:
    short = {
        "retrieve_policy":   "policy",
        "recommend_courses": "courses",
        "generate_assessment": "assess",
        None:                "(none)",
    }
    return short.get(name, str(name))

def _print_report(results: list) -> None:
    width = 80
    now   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print()
    print("═" * width)
    print(f"  {_BOLD}HR AVATAR EVALUATION REPORT{_RESET}  —  {now}")
    print("═" * width)

    # ── Per-case table ────────────────────────────────────────────────────────
    print()
    hdr = f"  {'#':<3} {'Test Case':<42} {'Expected':<9} {'Called':<9} {'Route':<7} {'Quality':<9} {'Time':>5}"
    print(hdr)
    print("  " + "─" * (width - 2))

    for i, r in enumerate(results, 1):
        guard_flag = f" {_YELLOW}[guard]{_RESET}" if r["hallucination_guard"] else ""
        row = (
            f"  {i:<3} "
            f"{r['name'][:41]:<42} "
            f"{_fmt_tool(r['expected_tool']):<9} "
            f"{_fmt_tool(r['tool_called']):<9} "
            f"{_tick(r['route_ok'])}      "
            f"{_tick(r['quality_ok'])}       "
            f"{r['latency']:>4.1f}s"
            f"{guard_flag}"
        )
        print(row)

    # ── Failures detail ───────────────────────────────────────────────────────
    failures = [r for r in results if not r["route_ok"] or not r["quality_ok"]]
    if failures:
        print()
        print(f"  {_BOLD}FAILURES{_RESET}")
        print("  " + "─" * (width - 2))
        for r in failures:
            print(f"\n  [{r['name']}]")
            print(f"    Input    : {r['input'][:70]}")
            print(f"    Expected : {r['expected_tool']}  →  Called: {r['tool_called']}")
            print(f"    Response : {r['response'][:120]}")

    # ── Summary ───────────────────────────────────────────────────────────────
    n         = len(results)
    route_n   = sum(r["route_ok"]   for r in results)
    quality_n = sum(r["quality_ok"] for r in results)
    pass_n    = sum(r["route_ok"] and r["quality_ok"] for r in results)
    guard_n   = sum(r["hallucination_guard"] for r in results)
    latencies = [r["latency"] for r in results]
    avg_lat   = statistics.mean(latencies)
    p95_lat   = sorted(latencies)[int(len(latencies) * 0.95)]

    def pct(k, total): return f"{k}/{total}  {100*k/total:5.1f}%"

    print()
    print(f"  {_BOLD}SUMMARY{_RESET}")
    print("  " + "─" * (width - 2))
    print(f"  {'Total cases':<28}: {n}")
    print(f"  {'Tool routing accuracy':<28}: {pct(route_n, n)}")
    print(f"  {'Response quality':<28}: {pct(quality_n, n)}")
    print(f"  {'Overall pass (both)':<28}: {pct(pass_n, n)}")
    print(f"  {'Hallucination guard fired':<28}: {guard_n}x")
    print(f"  {'Avg latency':<28}: {avg_lat:.1f}s")
    print(f"  {'P95 latency':<28}: {p95_lat:.1f}s")
    print()
    print("═" * width)
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _check_services()

    print(f"\nRunning {len(TEST_CASES)} test cases — this will take ~{len(TEST_CASES) * 5}–{len(TEST_CASES) * 10}s ...\n")

    results = []
    for i, case in enumerate(TEST_CASES, 1):
        print(f"  [{i:02}/{len(TEST_CASES)}] {case['name']} ...", end=" ", flush=True)
        r = _run_case(case)
        status = "PASS" if (r["route_ok"] and r["quality_ok"]) else "FAIL"
        colour = _GREEN if status == "PASS" else _RED
        print(f"{colour}{status}{_RESET}  ({r['latency']:.1f}s)")
        results.append(r)

    _print_report(results)
