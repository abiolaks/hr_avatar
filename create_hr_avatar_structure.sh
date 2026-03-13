#!/bin/bash

# Create the root directory
mkdir -p hr_avatar

# Navigate into it (optional, but helps for the rest)
cd hr_avatar

# Create subdirectories
mkdir -p assets brain vad voice face integration tests

# Create brain module files
touch brain/__init__.py
touch brain/rag.py          # RAGManager
touch brain/tools.py         # Tool definitions
touch brain/agent.py         # LangGraph agent
touch brain/state.py         # Custom state schema

# Create other module placeholders (you can add __init__.py if needed)
touch vad/__init__.py
touch voice/__init__.py
touch face/__init__.py
touch integration/__init__.py

# Create test files
touch tests/test_brain.py
touch tests/test_tools.py

# Create requirements.txt at root
touch requirements.txt

# Optional: Create a README.md
touch README.md

echo "Project structure created successfully under hr_avatar/"
