# mock_services.py
# Recommendation + assessment APIs backed by a real curated course catalog.
# Run on port 8001 in a separate terminal:
#   python mock_services.py
#
# Course URLs are real, free-to-access courses from fast.ai, Google, Kaggle,
# DeepLearning.AI, Harvard, freeCodeCamp, Coursera (free audit), and Microsoft.
# When the real LMS API is ready, swap RECOMMENDATION_API_URL in config.py —
# nothing else needs to change.

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title="Mock LMS Services")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── Real course catalog ───────────────────────────────────────────────────────
# Each entry: title, description, url, difficulty, duration_category, category, skills

COURSES = [
    # ── Python ──────────────────────────────────────────────────────────────
    {
        "title": "CS50P: Introduction to Programming with Python",
        "description": "Harvard's free Python course — from variables and loops to file I/O and unit tests. No prior experience needed.",
        "url": "https://cs50.harvard.edu/python/2022/",
        "difficulty": "Beginner",
        "duration_category": "Long",
        "category": "python",
        "skills": ["python", "programming", "scripting"],
    },
    {
        "title": "Kaggle Python",
        "description": "Hands-on Python exercises covering syntax, functions, loops, and list comprehensions — free and interactive.",
        "url": "https://www.kaggle.com/learn/python",
        "difficulty": "Beginner",
        "duration_category": "Short",
        "category": "python",
        "skills": ["python", "programming"],
    },
    {
        "title": "Automate the Boring Stuff with Python",
        "description": "Free online book and course — learn to automate real tasks: spreadsheets, PDFs, web scraping, and email.",
        "url": "https://automatetheboringstuff.com/",
        "difficulty": "Intermediate",
        "duration_category": "Medium",
        "category": "python",
        "skills": ["python", "automation", "scripting", "web scraping"],
    },
    {
        "title": "Python for Everybody Specialization",
        "description": "University of Michigan's beginner-to-intermediate Python series on Coursera — free to audit. Covers data structures, web APIs, and databases.",
        "url": "https://www.coursera.org/specializations/python",
        "difficulty": "Beginner",
        "duration_category": "Long",
        "category": "python",
        "skills": ["python", "data structures", "databases", "APIs"],
    },
    {
        "title": "Python Intermediate: Object-Oriented Programming",
        "description": "freeCodeCamp's free OOP course — classes, inheritance, encapsulation, and design patterns in Python.",
        "url": "https://www.freecodecamp.org/news/python-oop-full-course/",
        "difficulty": "Intermediate",
        "duration_category": "Short",
        "category": "python",
        "skills": ["python", "OOP", "design patterns"],
    },

    # ── Machine Learning ─────────────────────────────────────────────────────
    {
        "title": "Google Machine Learning Crash Course",
        "description": "Google's fast-paced ML intro with TensorFlow — covers linear models, neural networks, and real-world engineering practices. Free.",
        "url": "https://developers.google.com/machine-learning/crash-course",
        "difficulty": "Beginner",
        "duration_category": "Short",
        "category": "machine learning",
        "skills": ["machine learning", "TensorFlow", "neural networks", "supervised learning"],
    },
    {
        "title": "Kaggle: Intro to Machine Learning",
        "description": "Build your first ML models with scikit-learn in interactive notebooks. Decision trees, validation, and underfitting/overfitting explained.",
        "url": "https://www.kaggle.com/learn/intro-to-machine-learning",
        "difficulty": "Beginner",
        "duration_category": "Short",
        "category": "machine learning",
        "skills": ["machine learning", "scikit-learn", "decision trees"],
    },
    {
        "title": "Microsoft ML for Beginners",
        "description": "Microsoft's open-source 12-week curriculum on classical ML — regression, classification, clustering, NLP, and time series. Free on GitHub.",
        "url": "https://github.com/microsoft/ML-For-Beginners",
        "difficulty": "Beginner",
        "duration_category": "Long",
        "category": "machine learning",
        "skills": ["machine learning", "regression", "classification", "NLP"],
    },
    {
        "title": "Machine Learning Specialization — Andrew Ng",
        "description": "The gold-standard ML course: supervised learning, neural networks, recommender systems, and reinforcement learning. Free to audit on Coursera.",
        "url": "https://www.coursera.org/specializations/machine-learning-introduction",
        "difficulty": "Intermediate",
        "duration_category": "Long",
        "category": "machine learning",
        "skills": ["machine learning", "supervised learning", "neural networks", "reinforcement learning"],
    },
    {
        "title": "Kaggle: Intermediate Machine Learning",
        "description": "Go beyond the basics — missing values, categorical encoding, pipelines, cross-validation, XGBoost, and data leakage.",
        "url": "https://www.kaggle.com/learn/intermediate-machine-learning",
        "difficulty": "Intermediate",
        "duration_category": "Short",
        "category": "machine learning",
        "skills": ["machine learning", "XGBoost", "feature engineering", "pipelines"],
    },
    {
        "title": "fast.ai: Practical Machine Learning for Coders",
        "description": "Top-down, code-first approach to ML and deep learning. Train state-of-the-art models with minimal code. Completely free.",
        "url": "https://course.fast.ai/",
        "difficulty": "Intermediate",
        "duration_category": "Long",
        "category": "machine learning",
        "skills": ["machine learning", "deep learning", "PyTorch", "computer vision", "NLP"],
    },
    {
        "title": "Stanford CS229: Machine Learning",
        "description": "Stanford's rigorous ML lecture series — the mathematical foundations behind supervised, unsupervised, and reinforcement learning. Free on YouTube.",
        "url": "https://cs229.stanford.edu/",
        "difficulty": "Advanced",
        "duration_category": "Long",
        "category": "machine learning",
        "skills": ["machine learning", "probability", "optimisation", "SVMs", "EM algorithm"],
    },

    # ── Deep Learning ─────────────────────────────────────────────────────────
    {
        "title": "Deep Learning Specialization — Andrew Ng",
        "description": "Five courses covering neural networks, CNNs, sequence models, and ML strategy. Free to audit on Coursera.",
        "url": "https://www.coursera.org/specializations/deep-learning",
        "difficulty": "Intermediate",
        "duration_category": "Long",
        "category": "deep learning",
        "skills": ["deep learning", "CNNs", "RNNs", "TensorFlow", "Keras"],
    },
    {
        "title": "MIT 6.S191: Introduction to Deep Learning",
        "description": "MIT's annual deep learning bootcamp — lectures, labs, and projects covering the full modern DL stack. Free on MIT OpenCourseWare.",
        "url": "http://introtodeeplearning.com/",
        "difficulty": "Intermediate",
        "duration_category": "Medium",
        "category": "deep learning",
        "skills": ["deep learning", "TensorFlow", "computer vision", "generative models"],
    },
    {
        "title": "fast.ai: Part 2 — Deep Learning from the Foundations",
        "description": "Build a deep learning library from scratch in PyTorch. For practitioners who want to understand what happens under the hood.",
        "url": "https://course.fast.ai/Lessons/part2.html",
        "difficulty": "Advanced",
        "duration_category": "Long",
        "category": "deep learning",
        "skills": ["deep learning", "PyTorch", "backpropagation", "optimisers"],
    },

    # ── AI Agents ─────────────────────────────────────────────────────────────
    {
        "title": "Introduction to AI Agents — DeepLearning.AI",
        "description": "Beginner-friendly overview of what AI agents are, how they reason and use tools, and where they're applied today. Free short course.",
        "url": "https://www.deeplearning.ai/short-courses/ai-agents-in-langgraph/",
        "difficulty": "Beginner",
        "duration_category": "Short",
        "category": "AI agents",
        "skills": ["AI agents", "LLMs", "tool use", "reasoning"],
    },
    {
        "title": "AI Agents in LangGraph",
        "description": "Build multi-step AI agents with LangGraph — memory, tool use, cycles, and human-in-the-loop. Free short course by DeepLearning.AI.",
        "url": "https://www.deeplearning.ai/short-courses/ai-agents-in-langgraph/",
        "difficulty": "Intermediate",
        "duration_category": "Short",
        "category": "AI agents",
        "skills": ["AI agents", "LangGraph", "LangChain", "tool use", "memory"],
    },
    {
        "title": "LangChain for LLM Application Development",
        "description": "Learn to build LLM-powered apps using LangChain — chains, agents, memory, and retrieval-augmented generation. Free by DeepLearning.AI.",
        "url": "https://www.deeplearning.ai/short-courses/langchain-for-llm-application-development/",
        "difficulty": "Intermediate",
        "duration_category": "Short",
        "category": "AI agents",
        "skills": ["LangChain", "LLMs", "RAG", "agents", "prompt engineering"],
    },
    {
        "title": "Functions, Tools and Agents with LangChain",
        "description": "Master OpenAI function calling, tool use, and autonomous agent loops in LangChain. Free DeepLearning.AI short course.",
        "url": "https://www.deeplearning.ai/short-courses/functions-tools-agents-langchain/",
        "difficulty": "Intermediate",
        "duration_category": "Short",
        "category": "AI agents",
        "skills": ["function calling", "tool use", "agents", "LangChain"],
    },
    {
        "title": "Building Agentic RAG with LlamaIndex",
        "description": "Build RAG pipelines that use agents to reason across documents, route queries, and handle complex multi-step retrieval. Free short course.",
        "url": "https://www.deeplearning.ai/short-courses/building-agentic-rag-with-llamaindex/",
        "difficulty": "Advanced",
        "duration_category": "Short",
        "category": "AI agents",
        "skills": ["RAG", "LlamaIndex", "agents", "retrieval", "LLMs"],
    },

    # ── Data Science ──────────────────────────────────────────────────────────
    {
        "title": "Kaggle: Pandas",
        "description": "Learn to manipulate data with Pandas — indexing, groupby, merging, and reshaping DataFrames. Free and interactive.",
        "url": "https://www.kaggle.com/learn/pandas",
        "difficulty": "Beginner",
        "duration_category": "Short",
        "category": "data science",
        "skills": ["pandas", "python", "data analysis"],
    },
    {
        "title": "Kaggle: Data Visualisation",
        "description": "Create insightful charts with Seaborn — line charts, heatmaps, scatter plots, and choosing the right visual. Free.",
        "url": "https://www.kaggle.com/learn/data-visualization",
        "difficulty": "Beginner",
        "duration_category": "Short",
        "category": "data science",
        "skills": ["data visualisation", "seaborn", "matplotlib", "python"],
    },
    {
        "title": "Data Analysis with Python — freeCodeCamp",
        "description": "Full curriculum: NumPy, Pandas, Matplotlib, and five real-world data analysis projects. Free certification.",
        "url": "https://www.freecodecamp.org/learn/data-analysis-with-python/",
        "difficulty": "Intermediate",
        "duration_category": "Long",
        "category": "data science",
        "skills": ["python", "numpy", "pandas", "matplotlib", "data analysis"],
    },
    {
        "title": "Kaggle: Feature Engineering",
        "description": "Create better features — mutual information, clustering as features, PCA, and target encoding. Free Kaggle course.",
        "url": "https://www.kaggle.com/learn/feature-engineering",
        "difficulty": "Intermediate",
        "duration_category": "Short",
        "category": "data science",
        "skills": ["feature engineering", "machine learning", "PCA", "pandas"],
    },
]


# ── Keyword scoring ───────────────────────────────────────────────────────────
def _score(course: dict, keywords: list[str]) -> int:
    """Count how many keywords appear in the course's searchable fields."""
    haystack = " ".join([
        course["title"],
        course["description"],
        course["category"],
        " ".join(course["skills"]),
    ]).lower()
    return sum(1 for kw in keywords if kw in haystack)


# ── Assessments ───────────────────────────────────────────────────────────────
ASSESSMENTS = {

    # ── Python ────────────────────────────────────────────────────────────────

    # CS50P: Introduction to Programming with Python
    "cs50p": [
        {"question": "In Python, what does the 'pass' statement do?",
         "options": ["Acts as a no-op placeholder", "Ends the program", "Skips to the next loop iteration", "Raises an exception"]},
        {"question": "Which built-in function reads a line of text from the user?",
         "options": ["input()", "read()", "scan()", "get()"]},
        {"question": "What is the output of print(type(3.14))?",
         "options": ["<class 'float'>", "<class 'int'>", "<class 'double'>", "<class 'decimal'>"]},
    ],

    # Kaggle Python
    "kaggle-python": [
        {"question": "What does a list comprehension like [x*2 for x in range(3)] produce?",
         "options": ["[0, 2, 4]", "[1, 2, 3]", "[2, 4, 6]", "[0, 1, 2]"]},
        {"question": "Which method adds an element to the end of a list?",
         "options": ["append()", "add()", "insert()", "push()"]},
        {"question": "What does the 'in' operator check in Python?",
         "options": ["Membership in a sequence or collection", "Integer division", "Type equality", "Variable scope"]},
    ],

    # Automate the Boring Stuff with Python
    "automate-the-boring-stuff": [
        {"question": "Which module is used to work with file paths across operating systems?",
         "options": ["os.path / pathlib", "sys", "shutil", "glob"]},
        {"question": "What does the 're' module provide?",
         "options": ["Regular expression matching", "Remote execution", "File reading utilities", "Math functions"]},
        {"question": "Which library is commonly used for web scraping in Python?",
         "options": ["BeautifulSoup", "Flask", "Pandas", "Tkinter"]},
    ],

    # Python for Everybody Specialization
    "python-for-everybody": [
        {"question": "Which Python data structure stores key-value pairs?",
         "options": ["Dictionary", "List", "Tuple", "Set"]},
        {"question": "What does the requests library do?",
         "options": ["Makes HTTP calls to web APIs", "Handles database connections", "Reads CSV files", "Manages threads"]},
        {"question": "In SQL, which clause filters rows returned by a SELECT statement?",
         "options": ["WHERE", "HAVING", "GROUP BY", "ORDER BY"]},
    ],

    # Python Intermediate: Object-Oriented Programming
    "python-oop": [
        {"question": "What is the purpose of the '__init__' method in a Python class?",
         "options": ["Initialises a new instance's attributes", "Deletes an instance", "Defines class-level variables", "Imports modules"]},
        {"question": "Which OOP principle means a subclass can be used wherever a parent class is expected?",
         "options": ["Polymorphism", "Encapsulation", "Abstraction", "Composition"]},
        {"question": "What does the 'super()' function do?",
         "options": ["Calls a method from the parent class", "Creates a new subclass", "Overrides all parent methods", "Checks the class hierarchy"]},
    ],

    # ── Machine Learning ──────────────────────────────────────────────────────

    # kept for backwards-compatibility
    "python-101": [
        {"question": "What keyword defines a function in Python?",
         "options": ["def", "function", "fn", "lambda"]},
        {"question": "Which data type is mutable?",
         "options": ["tuple", "string", "list", "int"]},
        {"question": "What does len([1, 2, 3]) return?",
         "options": ["2", "3", "4", "0"]},
    ],
    "ml-101": [
        {"question": "What is overfitting?",
         "options": ["Good train / poor test performance", "Model too simple", "Too few parameters", "Slow training"]},
        {"question": "Which metric suits imbalanced classification best?",
         "options": ["Accuracy", "F1 Score", "MSE", "R-squared"]},
        {"question": "What does cross-validation help you estimate?",
         "options": ["Training speed", "Generalisation error", "Model size", "Dataset balance"]},
    ],

    # Google Machine Learning Crash Course
    "google-ml-crash-course": [
        {"question": "What is a feature in machine learning?",
         "options": ["An input variable used to make predictions", "The model's output", "A training algorithm", "A layer in a neural network"]},
        {"question": "What does gradient descent minimise?",
         "options": ["The loss function", "The number of features", "Training time", "Model size"]},
        {"question": "In TensorFlow, what is a tensor?",
         "options": ["A multi-dimensional array of data", "A single numerical value", "A type of neural network layer", "A training loop"]},
    ],

    # Kaggle: Intro to Machine Learning
    "kaggle-intro-to-machine-learning": [
        {"question": "What is a decision tree?",
         "options": ["A model that splits data by feature thresholds to make predictions", "A clustering algorithm", "A type of neural network", "A data preprocessing step"]},
        {"question": "What does 'validation set' mean?",
         "options": ["Data held out to tune and evaluate the model during training", "The full training dataset", "The final test data", "A set of hyperparameters"]},
        {"question": "What is underfitting?",
         "options": ["Model is too simple to capture patterns in the data", "Model memorises training data", "Model has too many parameters", "Model trains too slowly"]},
    ],

    # Microsoft ML for Beginners
    "microsoft-ml-for-beginners": [
        {"question": "Which type of learning uses labelled training examples?",
         "options": ["Supervised learning", "Unsupervised learning", "Reinforcement learning", "Self-supervised learning"]},
        {"question": "What does a clustering algorithm do?",
         "options": ["Groups similar data points without labels", "Predicts a continuous value", "Classifies text sentiment", "Reduces dimensionality"]},
        {"question": "Which algorithm is commonly used for time-series forecasting?",
         "options": ["ARIMA", "K-Means", "Random Forest", "PCA"]},
    ],

    # Machine Learning Specialization — Andrew Ng
    "machine-learning-specialization": [
        {"question": "What is regularisation used for in ML?",
         "options": ["Reducing overfitting by penalising large weights", "Speeding up training", "Increasing model capacity", "Normalising input features"]},
        {"question": "In collaborative filtering, what does the model learn?",
         "options": ["Latent features of users and items to predict preferences", "Image classifications", "Speech-to-text mappings", "Regression coefficients"]},
        {"question": "What is the main idea behind reinforcement learning?",
         "options": ["An agent learns by receiving rewards and penalties from an environment", "Training on labelled datasets", "Clustering unlabelled data", "Reducing dimensionality"]},
    ],

    # Kaggle: Intermediate Machine Learning
    "kaggle-intermediate-machine-learning": [
        {"question": "What does XGBoost stand for?",
         "options": ["Extreme Gradient Boosting", "Extended Graph Boost", "Extra Greedy Boost", "Explicit Gradient Boost"]},
        {"question": "How do you handle categorical features in scikit-learn pipelines?",
         "options": ["Use OrdinalEncoder or OneHotEncoder inside a ColumnTransformer", "Drop them automatically", "Convert to float manually", "Use only numeric features"]},
        {"question": "What is data leakage?",
         "options": ["When future information unintentionally influences the model during training", "When data is lost during preprocessing", "When the model leaks weights", "When training data is too small"]},
    ],

    # fast.ai: Practical Machine Learning for Coders
    "fastai-practical-machine-learning": [
        {"question": "What is the key principle of the fast.ai top-down teaching approach?",
         "options": ["Build working models first, then understand the theory", "Learn all maths before writing code", "Start with linear models only", "Use only pre-built APIs"]},
        {"question": "In PyTorch, what does loss.backward() do?",
         "options": ["Computes gradients via backpropagation", "Updates model weights", "Resets the optimiser", "Loads a checkpoint"]},
        {"question": "What is transfer learning?",
         "options": ["Starting from a pre-trained model and fine-tuning on a new task", "Transferring data between servers", "Moving a model from GPU to CPU", "Copying weights between two identical models"]},
    ],

    # Stanford CS229: Machine Learning
    "stanford-cs229": [
        {"question": "What does the EM algorithm stand for?",
         "options": ["Expectation-Maximisation", "Error Minimisation", "Entropy Measurement", "Estimate and Match"]},
        {"question": "What is the kernel trick in SVMs?",
         "options": ["It computes dot products in a high-dimensional space without explicit transformation", "It reduces the number of support vectors", "It regularises the margin", "It speeds up gradient descent"]},
        {"question": "What does a Gaussian Naive Bayes classifier assume about features?",
         "options": ["Features are conditionally independent and normally distributed given the class", "Features are correlated", "All features have equal variance", "Features must be binary"]},
    ],

    # ── Deep Learning ─────────────────────────────────────────────────────────

    # Deep Learning Specialization — Andrew Ng
    "deep-learning-specialization": [
        {"question": "What does a ReLU activation function return for a negative input?",
         "options": ["0", "The input value unchanged", "-1", "A random small value"]},
        {"question": "What is batch normalisation used for?",
         "options": ["Stabilising and accelerating training by normalising layer inputs", "Reducing the number of parameters", "Adding dropout regularisation", "Initialising weights"]},
        {"question": "In an LSTM, what is the purpose of the forget gate?",
         "options": ["Decides what information to discard from the cell state", "Adds new information to the cell state", "Produces the output", "Controls the learning rate"]},
    ],

    # MIT 6.S191: Introduction to Deep Learning
    "mit-intro-deep-learning": [
        {"question": "What problem does the vanishing gradient make difficult to train?",
         "options": ["Very deep networks", "Wide shallow networks", "Decision trees", "K-Means clustering"]},
        {"question": "What is a GAN?",
         "options": ["A framework where a generator and discriminator compete to produce realistic data", "A graph attention network", "A gated activation node", "A generalised additive model"]},
        {"question": "What does dropout do during training?",
         "options": ["Randomly sets neuron activations to zero to prevent co-adaptation", "Removes underperforming layers", "Resets the learning rate", "Prunes the smallest weights"]},
    ],

    # fast.ai: Part 2 — Deep Learning from the Foundations
    "fastai-deep-learning-foundations": [
        {"question": "What is the chain rule used for in backpropagation?",
         "options": ["Computing gradients through composed functions layer by layer", "Chaining multiple datasets together", "Linking optimisers to loss functions", "Scheduling the learning rate"]},
        {"question": "What does a custom PyTorch DataLoader need to implement?",
         "options": ["__len__ and __getitem__ methods in its Dataset", "A forward() method", "A loss function", "An optimiser step"]},
        {"question": "What is weight decay in optimisation?",
         "options": ["An L2 regularisation term that shrinks weights toward zero", "Reducing the learning rate over time", "Removing neurons with small weights", "A momentum-based update rule"]},
    ],

    # ── AI Agents ─────────────────────────────────────────────────────────────

    # Introduction to AI Agents — DeepLearning.AI
    "introduction-to-ai-agents": [
        {"question": "What makes an LLM-based system an 'agent' rather than a simple chatbot?",
         "options": ["It can take actions and use tools to affect its environment", "It responds faster", "It uses a larger language model", "It stores conversation history"]},
        {"question": "What is a 'tool' in the context of AI agents?",
         "options": ["A function the agent can call to retrieve information or perform actions", "A pre-trained model component", "A prompt template", "A type of memory store"]},
        {"question": "What is the ReAct framework?",
         "options": ["Interleaving reasoning steps and actions so the agent thinks before it acts", "A React.js front-end for AI apps", "A reinforcement learning algorithm", "A retrieval-augmented generation technique"]},
    ],

    # AI Agents in LangGraph
    "ai-agents-in-langgraph": [
        {"question": "What is a LangGraph 'node'?",
         "options": ["A function that receives and returns the graph state", "A database record", "A prompt template", "An API endpoint"]},
        {"question": "What does 'human-in-the-loop' mean in a LangGraph workflow?",
         "options": ["The graph pauses for a human to review or approve before continuing", "A human writes the graph code", "The agent asks the user for their name", "Manual data labelling"]},
        {"question": "How does LangGraph represent agent memory across turns?",
         "options": ["As a typed state object passed through the graph", "In a global Python variable", "In a SQL database by default", "As a JSON file on disk"]},
    ],

    # LangChain for LLM Application Development
    "langchain-for-llm-application-development": [
        {"question": "What is a LangChain 'chain'?",
         "options": ["A sequence of components (prompts, models, parsers) connected together", "A blockchain data structure", "A list of API keys", "A type of vector database"]},
        {"question": "What does a retrieval-augmented generation (RAG) chain do?",
         "options": ["Fetches relevant documents and injects them into the prompt before the LLM answers", "Generates new training data", "Retrieves model weights from a server", "Augments the training dataset"]},
        {"question": "What is LangChain Memory used for?",
         "options": ["Persisting conversation history so the LLM can reference earlier turns", "Caching API responses", "Storing model weights", "Compressing prompts"]},
    ],

    # Functions, Tools and Agents with LangChain
    "functions-tools-and-agents-with-langchain": [
        {"question": "What is OpenAI function calling?",
         "options": ["A protocol where the model returns structured JSON describing a function to call", "Calling Python functions inside the prompt", "An API for serverless functions", "A way to fine-tune the model"]},
        {"question": "What does bind_tools() do in LangChain?",
         "options": ["Attaches tool schemas to the model so it knows what tools are available", "Binds Python functions to keyboard shortcuts", "Links two LangChain chains together", "Registers tools in a database"]},
        {"question": "In an agent loop, what triggers the agent to stop?",
         "options": ["The model returns a final answer with no further tool calls", "A timeout", "The user closes the browser", "The context window is full"]},
    ],

    # Building Agentic RAG with LlamaIndex
    "building-agentic-rag-with-llamaindex": [
        {"question": "What is agentic RAG compared to standard RAG?",
         "options": ["The agent decides how and when to retrieve, using multi-step reasoning", "RAG with a larger vector store", "RAG that runs on a GPU", "RAG with streaming output"]},
        {"question": "What is a LlamaIndex 'query engine'?",
         "options": ["An interface that accepts a natural-language question and returns a grounded answer", "A SQL query builder", "A web scraper", "A model fine-tuning tool"]},
        {"question": "What does a router in LlamaIndex do?",
         "options": ["Selects which index or tool to use based on the query", "Routes HTTP requests", "Balances GPU load", "Schedules indexing jobs"]},
    ],

    # ── Data Science ──────────────────────────────────────────────────────────

    # Kaggle: Pandas
    "kaggle-pandas": [
        {"question": "What does df.groupby('col').mean() do?",
         "options": ["Groups rows by 'col' and computes the mean of each group", "Sorts the DataFrame by 'col'", "Filters rows where 'col' equals the mean", "Renames 'col' to 'mean'"]},
        {"question": "How do you select rows where column 'age' is greater than 30?",
         "options": ["df[df['age'] > 30]", "df.filter(age > 30)", "df.select('age' > 30)", "df.query.age(30)"]},
        {"question": "What does df.merge() do?",
         "options": ["Joins two DataFrames on a common key, similar to a SQL JOIN", "Concatenates DataFrames vertically", "Removes duplicate rows", "Sorts and resets the index"]},
    ],

    # Kaggle: Data Visualisation
    "kaggle-data-visualisation": [
        {"question": "Which Seaborn function creates a heatmap?",
         "options": ["sns.heatmap()", "sns.scatter()", "sns.barplot()", "sns.lineplot()"]},
        {"question": "When should you use a bar chart instead of a line chart?",
         "options": ["When comparing discrete categories rather than showing a trend over time", "When plotting continuous time-series data", "When showing correlations between two numeric variables", "When displaying a distribution of values"]},
        {"question": "What does a scatter plot help you identify?",
         "options": ["Correlations and relationships between two numeric variables", "The frequency of categories", "Changes over time", "Parts of a whole"]},
    ],

    # Data Analysis with Python — freeCodeCamp
    "data-analysis-with-python": [
        {"question": "What does NumPy's np.mean() compute?",
         "options": ["The arithmetic average of array elements", "The middle value of sorted elements", "The most frequent element", "The sum of elements"]},
        {"question": "Which Matplotlib function saves a figure to a file?",
         "options": ["plt.savefig()", "plt.save()", "plt.export()", "plt.write()"]},
        {"question": "What is the purpose of df.dropna() in Pandas?",
         "options": ["Removes rows (or columns) that contain missing values", "Fills missing values with zero", "Replaces NaN with the column mean", "Drops the last column"]},
    ],

    # Kaggle: Feature Engineering
    "kaggle-feature-engineering": [
        {"question": "What does mutual information measure?",
         "options": ["The statistical dependence between a feature and the target variable", "The correlation between two features", "The variance of a feature", "The number of unique values in a feature"]},
        {"question": "What is target encoding?",
         "options": ["Replacing a categorical value with the mean of the target for that category", "One-hot encoding the target variable", "Hashing categorical features", "Binning continuous features"]},
        {"question": "What does PCA do to a dataset?",
         "options": ["Reduces dimensionality by projecting data onto principal components", "Removes outliers", "Normalises feature scales", "Encodes categorical variables"]},
    ],
}


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.post("/recommend")
def recommend(payload: dict):
    difficulty      = payload.get("preferred_difficulty") or payload.get("skill_level", "Intermediate")
    duration        = payload.get("preferred_duration")
    learning_goal   = payload.get("learning_goal", "").lower()
    category        = payload.get("preferred_category", "").lower()
    enrolled        = [c.lower() for c in payload.get("enrolled_courses", [])]

    # Build keyword list from learning goal + category
    keywords = [w for w in (learning_goal + " " + category).split() if len(w) > 2]

    # Filter by difficulty; ignore duration (use it only as a tie-break score).
    candidates = [
        c for c in COURSES
        if c["difficulty"] == difficulty
        and c["title"].lower() not in enrolled
    ]

    # Fall back to all difficulties if strict filter leaves fewer than 3
    if len(candidates) < 3:
        candidates = [c for c in COURSES if c["title"].lower() not in enrolled]

    # Rank by keyword relevance
    candidates.sort(key=lambda c: _score(c, keywords), reverse=True)

    top3 = candidates[:3]
    return {"courses": [
        {"title": c["title"], "description": c["description"], "url": c["url"]}
        for c in top3
    ]}


@app.post("/generate")
def generate(payload: dict):
    course_id = payload.get("course_id", "").lower().replace(" ", "-")
    questions = ASSESSMENTS.get(course_id)

    if not questions:
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
