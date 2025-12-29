# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a RAG (Retrieval-Augmented Generation) chatbot for educational course materials. Users ask questions about course content and receive AI-powered answers with source citations.

## Development Guidelines

- Always use `uv` to run the app, do not use pip directly

## Commands

**Install dependencies:**
```bash
uv sync
```

**Run the application:**
```bash
cd backend && uv run uvicorn app:app --reload --port 8000
```

Or use the shell script (Linux/Mac):
```bash
./run.sh
```

**Access the app:** http://localhost:8000

## Architecture

```
Frontend (vanilla JS) → FastAPI → RAGSystem → Claude API
                                      ↓
                              ChromaDB (vectors)
```

### Query Flow

1. User submits question via web UI
2. `app.py` receives POST to `/api/query`
3. `RAGSystem.query()` invokes Claude with the `search_course_content` tool
4. Claude decides whether to search or answer directly
5. If searching: `CourseSearchTool` queries ChromaDB for relevant chunks
6. Claude synthesizes final answer from search results
7. Response + sources returned to frontend

### Key Components

| File | Purpose |
|------|---------|
| `backend/rag_system.py` | Main orchestrator - coordinates all components |
| `backend/ai_generator.py` | Claude API integration with tool handling |
| `backend/vector_store.py` | ChromaDB wrapper for semantic search |
| `backend/document_processor.py` | Parses course docs into chunks |
| `backend/search_tools.py` | Defines `search_course_content` tool for Claude |
| `backend/session_manager.py` | Conversation history per session |
| `backend/config.py` | Central configuration (chunk size, models, etc.) |

### Document Processing

Course documents in `docs/` follow this format:
```
Course Title: [title]
Course Link: [url]
Course Instructor: [name]

Lesson 0: [title]
[content...]

Lesson 1: [title]
[content...]
```

Documents are chunked (~800 chars with 100 char overlap) and embedded using SentenceTransformers (`all-MiniLM-L6-v2`).

### Two ChromaDB Collections

- `course_catalog` - Course metadata (titles, instructors)
- `course_content` - Actual text chunks for semantic search

## Configuration

Key settings in `backend/config.py`:
- `CHUNK_SIZE`: 800 chars
- `CHUNK_OVERLAP`: 100 chars
- `MAX_RESULTS`: 5 search results
- `ANTHROPIC_MODEL`: claude-sonnet-4-20250514

## Environment

Requires `.env` file with:
```
ANTHROPIC_API_KEY=your-key-here
```
