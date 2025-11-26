# 🧠 Financial Forecasting Agent

A FastAPI-based intelligent financial analysis system that **extracts PDF data**, **ingests it into a vector database**, and **generates a qualitative forecast** based on financial reports and earnings transcripts.

---

## 🚀 Project Overview

This project implements an **agentic reasoning pipeline** using **LLMs + tool calling**, designed to analyze financial documents and generate forecasts for the upcoming quarter.

### 🔍 Architectural Approach

```
User → FastAPI → Agent → PDF Extractor Tool → Text Splitter → Pinecone Vector DB →
→ Retrieval → Thought Chain → Financial Forecast Generation → Response
```

### 🧠 Agent Reasoning Method

The agent uses a **Chain-of-Thought + Tool Invocation** process:

1. Understands the user query.
2. Logs initial thought (`think` tool).
3. Attempts structured reasoning (`analyze` tool).
4. Extracts financial data using `financial_data_extractor`.
5. Retrieves qualitative insights using `qualitative_analysis`.
6. If retrieval fails → generates fallback analysis.
7. Logs the entire request + response to PostgreSQL.

---

## 🔧 Agent & Tool Design

| Tool Name | Purpose |
|-----------|---------|
| `think` | Logs internal reasoning step-by-step (not visible to user). |
| `analyze` | Breaks down user query logically. |
| `financial_data_extractor` | Retrieves structured metrics from PDF chunks (via vector DB). |
| `qualitative_analysis` | Retrieves earnings call-style commentary & management guidance. |

### 🧠 Master System Prompt (Agent Behavior)

```
You are a Financial Forecasting Agent.
Your job is to:
- Extract key quantitative trends from financial reports.
- Extract management outlook from transcripts.
- Identify risks and opportunities.
- Generate a human-like qualitative forecast.
You MUST chain thoughts first → then decide which tools to use.
If tools fail → still provide structured fallback analysis.
```

---

## ⚙️ Setup Instructions

### 1️⃣ Clone Repository
```bash
git clone <your-repo-url>
cd Financial-Forecasting-Agent
```

### 2️⃣ Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Configure .env
Create a `.env` file in root:
```env
# LLM Keys
OPENAI_API_KEY=your_openai_key
GROQ_API_KEY=your_groq_key

# Vector DB
PINECONE_API_KEY=your_pinecone_key
PINECONE_ENV=your_pinecone_env
PINECONE_INDEX=finance-agent-index

# MySQL (optional – currently not working)
MYSQL_HOST=localhost
MYSQL_USER=root
MYSQL_PASSWORD=your_password
MYSQL_DB=finance_agent

# PostgreSQL (WORKING)
PG_HOST=localhost
PG_USER=postgres
PG_PASSWORD=your_password
PG_DB=finance_agent_logs
PG_PORT=5432
```

---

## ⚡ How to Run the Agent

```bash
python -m uvicorn app:app --reload
```

Then open in browser:  
[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---
## 🏁 Conclusion

This system is a working LLM agent capable of:
- ✔ Extracting financial text
- ✔ Storing & retrieving via Pinecone
- ✔ Logging all interactions
- ✔ Generating structured forecasts

You are VERY close to a production-grade agent 🚀