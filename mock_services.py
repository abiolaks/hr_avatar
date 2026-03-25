# mock_services.py
# Fake recommendation + assessment APIs for demo/dev use.
# Run on port 8001 in a separate terminal:
#   python mock_services.py
#
# The real APIs will be built by the software engineering team.
# This file exists solely so the demo works end-to-end today.

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title="Mock LMS Services")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


COURSE_CATALOG = {
    "Beginner": [
        {"title": "Python for Absolute Beginners",    "description": "Hands-on intro to Python syntax, data types, and basic scripting."},
        {"title": "SQL Fundamentals",                 "description": "Write your first queries, JOINs, and aggregations in SQL."},
        {"title": "Excel for Data Analysis",          "description": "Pivot tables, VLOOKUP, and charts for everyday analytics."},
        {"title": "Intro to Statistics",              "description": "Mean, median, variance, and probability — no maths degree required."},
    ],
    "Intermediate": [
        {"title": "Machine Learning with scikit-learn", "description": "Train classifiers and regressors using real-world datasets."},
        {"title": "Data Visualisation with Python",     "description": "Create compelling charts with Matplotlib, Seaborn, and Plotly."},
        {"title": "Advanced SQL & Query Optimisation",  "description": "CTEs, window functions, and indexing strategies."},
        {"title": "Python for Data Engineering",        "description": "Pipelines, Pandas, and working with APIs and cloud storage."},
        {"title": "Statistics for Data Science",        "description": "Hypothesis testing, regression, and Bayesian thinking."},
    ],
    "Advanced": [
        {"title": "Deep Learning with PyTorch",        "description": "Build and train neural networks from scratch."},
        {"title": "MLOps & Model Deployment",          "description": "CI/CD for ML, Docker, and model serving with FastAPI."},
        {"title": "Large Language Models in Practice", "description": "Fine-tuning, RAG, and building LLM-powered applications."},
        {"title": "Distributed Data Processing",       "description": "Spark, Kafka, and streaming architectures at scale."},
    ],
}

WEEKEND_COURSES = [
    {"title": "Python in a Weekend",          "description": "Intensive crash course — go from zero to scripting in 2 days."},
    {"title": "SQL Bootcamp (Short Format)",  "description": "Cover all core SQL concepts in 4 focused sessions."},
    {"title": "Quick-Start: Data Analysis",   "description": "Pandas and Matplotlib essentials in under 6 hours total."},
]

ASSESSMENTS = {
    "python-101": [
        {"question": "What keyword is used to define a function in Python?",
         "options": ["def", "function", "fn", "lambda"]},
        {"question": "Which of the following is a mutable data type in Python?",
         "options": ["tuple", "string", "list", "int"]},
        {"question": "What does `len([1, 2, 3])` return?",
         "options": ["2", "3", "4", "0"]},
        {"question": "How do you open a file for reading in Python?",
         "options": ["open('file.txt', 'r')", "read('file.txt')", "file.open('r')", "load('file.txt')"]},
    ],
    "sql-101": [
        {"question": "Which SQL clause filters rows after aggregation?",
         "options": ["WHERE", "HAVING", "GROUP BY", "ORDER BY"]},
        {"question": "What does SELECT DISTINCT do?",
         "options": ["Sorts results", "Returns unique rows", "Counts rows", "Joins tables"]},
    ],
    "ml-basics": [
        {"question": "What is overfitting?",
         "options": ["Model performs well on training but poorly on new data",
                     "Model is too simple", "Model has too few parameters", "Model trains too slowly"]},
        {"question": "Which metric is best for imbalanced classification?",
         "options": ["Accuracy", "F1 Score", "MSE", "R-squared"]},
    ],
}


@app.post("/recommend")
def recommend(payload: dict):
    skill_level   = payload.get("preferred_difficulty") or payload.get("skill_level", "Intermediate")
    duration      = payload.get("preferred_duration", "Short")
    learning_goal = payload.get("learning_goal", "").lower()

    # Short duration → weekend-friendly courses
    if duration == "Short":
        courses = WEEKEND_COURSES.copy()
    else:
        courses = COURSE_CATALOG.get(skill_level, COURSE_CATALOG["Intermediate"]).copy()

    # Boost relevance: put courses that match keywords first
    keywords = learning_goal.split()
    def relevance(c):
        title_lower = c["title"].lower()
        return sum(k in title_lower for k in keywords)

    courses.sort(key=relevance, reverse=True)
    return {"courses": courses[:3]}


@app.post("/generate")
def generate(payload: dict):
    course_id = payload.get("course_id", "").lower().replace(" ", "-")
    questions = ASSESSMENTS.get(course_id)

    if not questions:
        # Generic fallback assessment
        questions = [
            {"question": f"What was the main concept covered in '{payload.get('course_id', 'this course')}'?",
             "options": ["Foundations and core principles", "Advanced applications", "Case studies only", "None of the above"]},
            {"question": "How confident are you in applying what you learned?",
             "options": ["Very confident", "Somewhat confident", "Need more practice", "Not confident"]},
            {"question": "Would you recommend this course to a colleague?",
             "options": ["Definitely", "Probably", "Not sure", "No"]},
        ]

    return {"questions": questions}


@app.get("/health")
def health():
    return {"status": "ok", "service": "mock-lms-services"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)
