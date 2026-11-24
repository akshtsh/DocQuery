# DocQuery – Conversational PDF Research Agent

## 1. Project Introduction

**DocQuery** is a beginner-friendly Conversational PDF Question Answering system built with **Retrieval-Augmented Generation (RAG)**, **Conversational Memory**, and **Agentic AI** using Python, LangChain, and the Groq API.

Users can upload any text-based PDF document and interact with it through a clean Streamlit chatbot. Instead of acting like a plain search wrapper, DocQuery uses a **LangChain tool-calling agent** that understands follow-up questions from recent conversation history, chooses the right tool (`search_pdf`, `summarize_topic`, or `compare_topics`), retrieves relevant chunks from ChromaDB with page references, verifies the answer against the retrieved text, and displays an expandable activity log.

---

## 2. Technologies Used

| Component | Technology | Role |
|---|---|---|
| Programming Language | Python 3.10+ | Core application logic |
| User Interface | Streamlit | Minimal web chatbot interface |
| Agent & RAG Framework | LangChain | Tool-calling agent, prompt orchestration, document splitting |
| LLM | Groq API (`openai/gpt-oss-120b` default) | Fast reasoning, tool selection, answer generation, verification |
| Embedding Model | `sentence-transformers/all-MiniLM-L6-v2` | Local 384-dimensional vector embeddings |
| Vector Database | ChromaDB (`chroma_db/`) | Persistent local vector storage and similarity retrieval |
| PDF Loader | `PyPDFLoader` (`pypdf`) | Page-by-page PDF text extraction with page metadata |
| Text Splitter | `RecursiveCharacterTextSplitter` | Splitting PDF pages into overlapping chunks |

> Note: Embeddings are generated **100% locally** using `sentence-transformers/all-MiniLM-L6-v2`. Groq is used **only** for LLM reasoning and answer generation.

---

## 3. System Architecture & Project Structure

```text
DocQuery/
│
├── app.py                  # Streamlit user interface
├── requirements.txt        # Python dependencies
├── .env.example            # Template for environment variables
├── .gitignore              # Git ignore rules (.env, chroma_db, etc.)
├── README.md               # Project documentation & viva Q&A
│
├── chroma_db/              # Local persistent ChromaDB storage
│
└── utils/
    ├── __init__.py
    ├── pdf_processor.py    # PDF validation, extraction, and chunking
    ├── vector_store.py     # Local All-MiniLM-L6-v2 embeddings & ChromaDB
    ├── memory.py           # Last-6-exchanges session state memory
    └── agent.py            # LangChain agent, 3 tools, ReAct log & verification
```

### Expected Workflow

```text
PDF Upload
  ↓
PDF Text Extraction (PyPDFLoader)
  ↓
Text Chunking (RecursiveCharacterTextSplitter: 1000 chars, 150 overlap)
  ↓
Local Embeddings (sentence-transformers/all-MiniLM-L6-v2)
  ↓
ChromaDB Storage (chroma_db/)

User Question
  ↓
Agent Receives Question + Last 6 Exchanges (Conversation History)
  ↓
Agent Understands User Intent (Reason)
  ↓
Agent Selects Appropriate Tool (Act: search_pdf / summarize_topic / compare_topics)
  ↓
Tool Executes ChromaDB Top-4 Retrieval (Observe)
  ↓
Answer Generation (Respond)
  ↓
Basic Answer Verification (Supported by PDF?)
  ↓
Final Response + Source Page References + Agent Activity Log
  ↓
Save Conversation in Streamlit Session State
```

---

## 4. Core Concepts Explained

### What is RAG (Retrieval-Augmented Generation)?
Large Language Models (LLMs) do not know the contents of private or newly uploaded documents. **RAG** solves this by first *retrieving* the most relevant passages from the user's PDF and passing those passages to the LLM as context so it can *generate* a grounded, accurate answer with page citations.

### What is `all-MiniLM-L6-v2`?
`sentence-transformers/all-MiniLM-L6-v2` is a fast, lightweight local embedding model from Hugging Face / Sentence Transformers. It converts text chunks and user queries into **384-dimensional dense vectors** that capture semantic meaning, enabling similarity search without calling any external embedding API.

### What is ChromaDB?
**ChromaDB** is an open-source vector database. DocQuery stores the PDF's text chunks, page metadata, and All-MiniLM-L6-v2 vectors in a local persistent directory (`chroma_db/`). When a user asks a question, ChromaDB quickly finds the top 4 most semantically similar chunks. Whenever a new PDF is uploaded, the previous collection is cleared so only the active PDF is searched.

### How Does Conversational Memory Work?
DocQuery uses **Streamlit session state** (`st.session_state`) to store the chat history and passes the **last 6 Q&A exchanges** (up to 12 messages) to the agent. This lets users ask natural follow-up questions like *"What are its three types?"* or *"Explain the first one with an example"* without repeating the topic. Clearing the chat removes the conversation history without deleting the PDF embeddings.

### What is Agentic AI & Tool Calling?
Unlike a fixed RAG pipeline that blindly runs the same retrieval step every time, **Agentic AI** allows the LLM to inspect the user's intent and dynamically choose what action to take. With **LangChain tool calling**, we register Python functions (`@tool`) with clear descriptions and parameters. The LLM outputs structured tool calls specifying which tool to invoke and with what arguments.

### What is the ReAct Workflow?
**ReAct** stands for **Reason → Act → Observe → Respond**:
1. **Reason:** Understand the user's question (plus conversation history) and decide whether to search, summarize, compare, or reply from existing context.
2. **Act:** Call the chosen tool (`search_pdf`, `summarize_topic`, or `compare_topics`).
3. **Observe:** Inspect the chunks and page numbers returned by ChromaDB.
4. **Respond:** Generate the final answer (using tables for comparisons, bullet points for summaries, and page citations), then run a simple verification check.

---

## 5. Explanation of Every Python File

- **`app.py`**: Builds the Streamlit web interface. Manages the sidebar (PDF uploader, filename, processing status, **Clear Chat**, **Upload Another PDF**), caches the embedding model, ensures embeddings are generated only once per uploaded PDF, displays chat messages, and renders expandable **Source Page References** and **Agent Activity** panels.
- **`utils/pdf_processor.py`**: Validates that the uploaded PDF is non-empty and valid, extracts page text with `PyPDFLoader`, preserves 1-based `page_number` and `source_file` metadata, rejects scanned/empty PDFs with a helpful error, and splits pages into chunks (`chunk_size=1000`, `chunk_overlap=150`) via `RecursiveCharacterTextSplitter`.
- **`utils/vector_store.py`**: Defines `LocalSentenceTransformerEmbeddings` wrapping `sentence-transformers/all-MiniLM-L6-v2` locally, replaces the active ChromaDB collection in `chroma_db/` when a new PDF is uploaded, loads the existing collection during normal conversation, and performs top-4 similarity search filtered by the active `doc_id`.
- **`utils/memory.py`**: Manages `chat_history` inside `st.session_state`. Stores messages along with their sources, activity logs, and verification results, and converts the last 6 Q&A pairs into LangChain `HumanMessage` and `AIMessage` objects.
- **`utils/agent.py`**: Registers the 3 PDF tools, builds the `create_tool_calling_agent` + `AgentExecutor`, records tool execution steps in a clean activity log (without exposing private chain-of-thought), and performs a single lightweight **Answer Verification** check to ensure responses are grounded in the PDF.

---

## 6. Explanation of Every Tool

| Tool Name | Arguments | Purpose |
|---|---|---|
| `search_pdf` | `query: str` | Searches ChromaDB for the top 4 chunks matching a factual question and returns the text with page numbers. |
| `summarize_topic` | `topic: str` | Retrieves the top 4 relevant PDF chunks for a requested topic so the agent can write a structured bullet-point summary. |
| `compare_topics` | `topic_a: str, topic_b: str` | Queries ChromaDB separately for both topics (top 4 chunks each) and returns both sets of passages so the agent can generate a Markdown comparison table. |

---

## 7. Step-by-Step Project Execution Flow

1. User launches the Streamlit app and opens it in a browser.
2. User uploads a PDF file in the sidebar.
3. `pdf_processor.py` validates the file, extracts text per page, and splits it into 1000-character chunks (150 overlap).
4. `vector_store.py` embeds the chunks locally using `all-MiniLM-L6-v2` and saves them in `chroma_db/`.
5. User types a question in the chat box.
6. `memory.py` fetches the last 6 Q&A exchanges from `st.session_state`.
7. `agent.py` passes the question and chat history to the LangChain tool-calling agent.
8. The agent chooses `search_pdf`, `summarize_topic`, or `compare_topics` (or replies from recent context if no lookup is needed) and observes the retrieved chunks.
9. The agent drafts an answer with page citations; `verify_answer()` checks that the answer is supported by the PDF and does not fabricate outside facts.
10. `app.py` displays the answer along with expandable **Source Page References** and **Agent Activity**, and saves the turn in memory.

---

## 8. Installation & Execution Instructions

### 1. Clone or Open the Project Folder
```bash
cd "PDF Q&A"
```

### 2. Create and Activate a Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env` and add your Groq API key:
```bash
cp .env.example .env
```
Edit `.env`:
```env
GROQ_API_KEY=gsk_your_actual_groq_key
GROQ_MODEL=openai/gpt-oss-120b
```

### 5. Run DocQuery
```bash
streamlit run app.py
```

---

## 9. 10 Basic Viva Questions & Answers

1. **What is the difference between a standard LLM and a RAG system?**
   A standard LLM relies only on its training data and cannot read your private PDFs. A RAG system retrieves relevant chunks from your PDF at query time and gives them to the LLM so answers are grounded in your document.

2. **Why did you use `RecursiveCharacterTextSplitter` with `chunk_size=1000` and `chunk_overlap=150`?**
   LLMs and embedding models work best on focused passages rather than entire books. A 1000-character chunk keeps paragraphs coherent, while a 150-character overlap prevents sentences or ideas from being cut awkwardly at chunk boundaries.

3. **Why use `sentence-transformers/all-MiniLM-L6-v2` instead of cloud embedding APIs?**
   `all-MiniLM-L6-v2` runs completely locally on CPU, costs nothing per token, keeps document embedding fast and private, and pairs cleanly with Groq (which provides fast LLM inference without an embeddings API).

4. **How do you ensure embeddings are not regenerated for every user question?**
   We compute a hash (`doc_id`) of the uploaded PDF and store it in `st.session_state`. Embeddings are created only when a new or different PDF is uploaded; subsequent questions reuse the active ChromaDB collection.

5. **How do you prevent chunks from a previously uploaded PDF from appearing in answers?**
   Whenever a different PDF is uploaded, `create_vector_store()` deletes the old active collection in ChromaDB and creates a fresh one tagged with the new `doc_id`, and all searches filter by `doc_id`.

6. **How does conversational memory help with follow-up questions?**
   `utils/memory.py` stores the last 6 user–assistant exchanges in `st.session_state` and passes them as `chat_history` to the agent. The agent uses this history to resolve pronouns like *"it"* or *"the first one"* into full topics before querying ChromaDB.

7. **Why did you use an agent with 3 tools instead of a single similarity search chain?**
   Different user requests need different retrieval strategies. For example, comparing two concepts requires searching ChromaDB separately for Topic A and Topic B (`compare_topics`), whereas a general question needs `search_pdf`, and a summary request uses `summarize_topic`.

8. **What is the ReAct workflow in DocQuery?**
   ReAct stands for Reason → Act → Observe → Respond. The agent reasons about the question and history, acts by calling a tool, observes the retrieved PDF chunks from ChromaDB, and responds with a grounded answer.

9. **What happens if a user asks something that is not in the uploaded PDF?**
   The retrieved chunks will not contain the answer, and both the agent instructions and the `verify_answer()` validation step ensure the system clearly tells the user that the information could not be found in the uploaded document.

10. **What is the difference between "Clear Chat" and "Upload Another PDF"?**
    **Clear Chat** only empties the conversation history in `st.session_state` while keeping the current PDF's embeddings intact in ChromaDB. **Upload Another PDF** resets the active document, clears the chat history, and lets the user index a new file.
