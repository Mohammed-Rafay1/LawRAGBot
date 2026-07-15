# ⚖️ Insaf-Guide — AI Legal Advisor for Pakistani Law

> An intelligent, RAG-powered legal advisor that provides accurate, cited legal guidance based on Pakistan's Constitution, PPC, CrPC, Family Laws, Corporate Law, Tax Law, and Supreme Court judgments.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green)
![LangGraph](https://img.shields.io/badge/LangGraph-Latest-orange)
![Qdrant](https://img.shields.io/badge/Qdrant-Local-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Legal Coverage](#legal-coverage)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Data & Vector Store Setup](#data--vector-store-setup)
- [Running the App](#running-the-app)
- [Usage Guide](#usage-guide)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [Disclaimer](#disclaimer)

---

## Overview

Insaf-Guide is a **Retrieval-Augmented Generation (RAG)** legal chatbot built specifically for Pakistani law. It combines a local vector database of 93,615+ legal chunks from statutes, court judgments, and Supreme Court case records with a powerful LLM to deliver lawyer-style responses — complete with section citations, court precedents, and actionable next steps.

Unlike generic AI assistants, Insaf-Guide:
- Cites **exact sections** from Pakistani statutes
- References **real Supreme Court judgments**
- Responds in **English, Urdu, or Roman Urdu** automatically
- Drafts **formal legal documents** (notices, applications, petitions)
- Remembers **conversation context** across multiple turns

---

## Features

| Feature | Description |
|---|---|
| 🔍 **RAG Pipeline** | LangGraph-powered Router → Retriever → Generator |
| 📚 **93,615+ vectors** | 36 statutes + 20 judgments + 1,414 HuggingFace SC cases |
| 🧠 **Conversation Memory** | Remembers full case context across multiple questions |
| 📝 **Document Drafting** | Generates legal notices, bail applications, petitions |
| 🌐 **Multilingual** | Auto-detects English, Urdu, Roman Urdu |
| 🕌 **RTL Support** | Proper Nastaliq font for Urdu responses |
| 📊 **Confidence Score** | High/Medium/Low based on retrieved chunks |
| 🏷️ **Category Routing** | 8 legal categories auto-classified |
| 🔒 **Private** | Runs 100% locally — no data sent to external servers |
| ⚡ **Fast** | Groq/Cerebras inference — responses in 3-8 seconds |

---

## Architecture

```
User Query
    │
    ▼
┌─────────────┐
│   Router    │  ← Classifies category (Criminal/Civil/Family etc.)
│   Node      │  ← Detects vague queries, draft requests
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Retriever  │  ← Semantic search on Qdrant (93,615 vectors)
│   Node      │  ← Enriches follow-up queries with history context
└──────┬──────┘
       │
       ├──── is_draft? ──▶ Drafter Node ──▶ Legal Document
       │
       ▼
┌─────────────┐
│  Generator  │  ← Lawyer-tone response with citations
│   Node      │  ← Uses statute + judgment chunks
└──────┬──────┘
       │
       ▼
  Legal Response
  (with section citations, court precedents, next steps)
```

### LLM Calls Per Query
- **Normal query**: 2 calls (Router + Generator)
- **Vague query**: 2 calls (Router + Clarifier)
- **Draft request**: 2 calls (Router + Drafter)

---

## Legal Coverage

### Statutes (36 PDFs)

| Category | Laws Covered |
|---|---|
| **Constitutional** | Constitution of Pakistan 1973, 27th Amendment 2025 |
| **Criminal** | PPC 1860, CrPC 1898, PECA 2025, Qanun-e-Shahadat 1984, Anti-Terrorism Act, Narcotic Substances Act, Hudood Ordinance |
| **Civil** | CPC 1908, Contract Act 1872, Specific Relief Act 1877, Transfer of Property Act, Limitation Act 1908, Registration Act, Punjab Tenancy Act, Land Revenue Act |
| **Family** | Muslim Family Laws Ordinance 1961, Family Courts Act 1964, Guardians & Wards Act, Child Marriage Restraint Act, Dowry Act |
| **Corporate** | Companies Act 2017, Partnership Act 1932, Negotiable Instruments Act, Sale of Goods Act |
| **Tax** | Income Tax Ordinance 2001, Sales Tax Act 1990, Finance Act 2025, Customs Act 1969 |
| **Labour** | Industrial Relations Act 2012, Factories Act 1934, Payment of Wages Act 1936 |
| **Shariah** | Federal Shariat Court Rules 1981 |

### Judgments (20 PDFs)
Supreme Court of Pakistan judgments on:
- Murder (Section 302 PPC)
- Khula & Divorce
- Inheritance
- Specific Performance
- Dissolution of Muslim Marriages

### HuggingFace Dataset
[`Ibtehaj10/supreme-court-of-pak-judgments`](https://huggingface.co/datasets/Ibtehaj10/supreme-court-of-pak-judgments) — 1,414 Supreme Court records (loaded automatically by the Kaggle notebook)

---

## Project Structure

```
insaf-guide/
├── src/
│   ├── main.py          # FastAPI backend
│   ├── graph.py         # LangGraph pipeline
│   └── data.py          # PDF ingestion & vector store builder
├── static/
│   ├── index.html       # Frontend UI
│   ├── style.css        # Dark theme CSS
│   └── script.js        # Chat logic + memory
├── data/
│   ├── raw/             # Statute PDFs (gitignored)
│   ├── judgments/       # Judgment PDFs (gitignored)
│   └── qdrant_db/       # Vector store (gitignored, ~200MB)
├── .env                 # Your API keys (gitignored)
├── .env.example         # Environment variable template
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Prerequisites

- Python 3.10 or higher
- 4GB+ RAM (for embedding model)
- 500MB+ disk space (for vector store)
- A free [Groq API key](https://console.groq.com) OR [Cerebras API key](https://cloud.cerebras.ai)

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/insaf-guide.git
cd insaf-guide
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Mac/Linux
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

```bash
cp .env.example .env
```

Edit `.env` and add your API key:

```env
GROQ_API_KEY=your_groq_key_here
```

Get a free key at [console.groq.com](https://console.groq.com) — 100,000 tokens/day free.

---

## Data & Vector Store Setup

The PDFs and vector store are **not included** in this repository due to file size. You have two options:

---

### ✅ Option A — Use Kaggle (Recommended, No Local Setup Needed)

All data is publicly hosted on Kaggle. You just need to run one notebook.

#### Step 1 — Download the PDFs (optional, for local use)

| Dataset | Contents | Link |
|---|---|---|
| 📄 **insaf-guide-pdfs** | 36 Pakistani statute PDFs | [kaggle.com/datasets/azeemnaseer/insaf-guide-pdfs](https://www.kaggle.com/datasets/azeemnaseer/insaf-guide-pdfs) |
| ⚖️ **insaf-guide-judgments** | 20 Supreme Court judgment PDFs | [kaggle.com/datasets/azeemnaseer/insaf-guide-judgments](https://www.kaggle.com/datasets/azeemnaseer/insaf-guide-judgments) |

If you want to use the PDFs locally, download both datasets and place:
- Statute PDFs → `data/raw/`
- Judgment PDFs → `data/judgments/`

#### Step 2 — Build the Vector Store on Kaggle (Free)

Use the public Kaggle notebook to build the full 93,615-vector Qdrant database for free:

🔗 **[insaf-guide-dataset-making](https://www.kaggle.com/code/azeemnaseer/insaf-guide-dataset-making)**

**Instructions:**

1. Open the notebook link above
2. Click **Copy & Edit** to fork it to your account
3. Attach both datasets to the notebook:
   - Click **Data** panel (right side) → **+ Add Data**
   - Search and attach `azeemnaseer/insaf-guide-pdfs`
   - Search and attach `azeemnaseer/insaf-guide-judgments`
4. Click **Run All** — takes approximately 60 minutes on Kaggle CPU
5. When finished, go to the **Output** tab and download `qdrant_db.zip`

> The notebook also automatically downloads the HuggingFace dataset `Ibtehaj10/supreme-court-of-pak-judgments` (1,414 SC records) — no extra step needed.

#### Step 3 — Extract the Vector Store Locally

```bash
unzip qdrant_db.zip -d data/
```

Your folder structure should look like:

```
data/
└── qdrant_db/
    ├── collection/
    └── meta.json
```

You're ready to run the app.

---

### 🔧 Option B — Build Locally from Scratch

1. Download PDFs from the Kaggle datasets above
2. Place statute PDFs in `data/raw/`
3. Place judgment PDFs in `data/judgments/`
4. Run the vector store builder:

```bash
python src/data.py
```

> ⚠️ This can take 30–90 minutes depending on your machine and number of PDFs.

---

## Running the App

Make sure your `.env` file has your API key, then:

```bash
python src/main.py
```

Or export the key directly:

```bash
export GROQ_API_KEY="your_key_here"   # Mac/Linux
set GROQ_API_KEY=your_key_here        # Windows

python src/main.py
```

Open your browser at **http://localhost:8000**

---

## Usage Guide

### Basic Legal Questions

Simply type your question in English, Urdu, or Roman Urdu:

```
What is the punishment for murder under Section 302 PPC?
مجھے طلاق لینی ہے، کیا کرنا ہوگا؟
Bail kaise milti hai Section 302 mein?
```

### Multi-turn Conversations

The bot remembers your full case context across messages:

```
You:  My employer has not paid my salary for 3 months
Bot:  [Full legal guidance on Payment of Wages Act...]

You:  Can I also claim damages on top of that?
Bot:  As you mentioned about your unpaid salary... [continues with damages]

You:  Draft a legal notice to my employer
Bot:  [Formal legal notice with placeholders]
```

### Document Drafting

Ask for any legal document:

```
Draft a legal notice to my landlord for return of security deposit
Draft a bail application under Section 497 CrPC
Draft a khula petition for the family court
Draft an affidavit for property transfer
```

### Category Examples

| Category | Example Query |
|---|---|
| 🚔 Criminal | "What is Section 420 PPC?" |
| 📋 Civil | "What is specific performance of a contract?" |
| 👨‍👩‍👧 Family | "How do I file for khula?" |
| 🏢 Corporate | "Duties of a director under Companies Act 2017?" |
| 📜 Constitutional | "What are fundamental rights under Article 9?" |
| 💰 Tax | "Income tax rates for salaried persons?" |
| 👷 Labour | "Can I file complaint for unpaid wages?" |
| 🕌 Shariah | "What is qatl-e-amd vs qatl-e-khata?" |

---

## Configuration

All configuration is in `src/graph.py`:

```python
GROQ_MODEL      = "llama-3.3-70b-versatile"   # LLM model
EMBEDDING_MODEL = "multi-qa-MiniLM-L6-cos-v1" # Embedding model
MAX_RETRIES     = 2                            # Max retrieval retries
```

### Switching to Cerebras (10x more free tokens — 1M/day)

```bash
pip install langchain-cerebras
export CEREBRAS_API_KEY="your_key"
```

In `src/graph.py`, replace the `get_llm()` function:

```python
from langchain_cerebras import ChatCerebras

def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatCerebras(
            model="llama-3.3-70b",
            api_key=os.environ.get("CEREBRAS_API_KEY", ""),
        )
    return _llm
```

### Adding New Laws

To add more PDFs to the knowledge base:

1. Add the PDF to `data/raw/`
2. Add an entry to `PDF_CATEGORY_MAP` in `src/data.py`
3. Add an entry to `PDF_LAW_NAMES` in `src/data.py`
4. Rebuild the vector store: `python src/data.py`

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | Yes (or Cerebras) | Get free key at [console.groq.com](https://console.groq.com) |
| `CEREBRAS_API_KEY` | Optional | Get free key at [cloud.cerebras.ai](https://cloud.cerebras.ai) — 1M tokens/day |

---

## API Reference

### POST `/api/chat`

Send a legal query and receive a response.

**Request:**
```json
{
  "message": "What is Section 302 PPC?",
  "chat_history": [
    "User: previous question",
    "Assistant: previous answer"
  ]
}
```

**Response:**
```json
{
  "response": "Under Section 302 of the PPC...",
  "category": "Criminal",
  "is_vague": false,
  "is_draft": false,
  "retries": 0,
  "confidence": "High"
}
```

### GET `/api/health`

```json
{
  "status": "ok",
  "service": "Insaf-Guide",
  "model": "llama-3.3-70b-versatile"
}
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Backend | FastAPI + Uvicorn |
| AI Pipeline | LangGraph |
| LLM | Groq (llama-3.3-70b-versatile) |
| Embeddings | sentence-transformers/multi-qa-MiniLM-L6-cos-v1 |
| Vector DB | Qdrant (local) |
| PDF Parsing | LangChain PyPDFLoader |
| Frontend | Vanilla HTML/CSS/JS |
| Markdown | marked.js |
| Fonts | Google Fonts (Inter, Outfit, Noto Nastaliq Urdu) |

---

## Contributing

Pull requests are welcome. For major changes please open an issue first.

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature/my-feature`
5. Open a pull request

---

## Disclaimer

> **Insaf-Guide provides informational guidance only — not legal advice.**
>
> This tool is designed to help users understand Pakistani law and legal procedures. It does not constitute legal advice and should not be relied upon as a substitute for professional legal counsel. Always consult a qualified and licensed advocate for formal legal representation.
>
> The information provided is based on publicly available legal texts and may not reflect the most recent amendments or judicial interpretations. Laws change frequently — always verify with current sources.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built with ❤️ for access to justice in Pakistan*