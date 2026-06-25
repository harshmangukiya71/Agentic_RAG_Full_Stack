# DocRAG - Hybrid Agentic RAG Assistant

## Live Demo

Frontend:
https://agentic-rag-full-stack.vercel.app/

Backend API:
https://agenticragfullstack-production.up.railway.app/docs



## Architecture

```text
                                        ┌─────────────────────────┐
                                        │      User Query         │
                                        └────────────┬────────────┘
                                                     │
                                                     ▼
                                        ┌─────────────────────────┐
                                        │     FastAPI Backend     │
                                        └────────────┬────────────┘
                                                     │
                                                     ▼
                                        ┌─────────────────────────┐
                                        │ Query Rewriter /        │
                                        │ Query Optimizer (LLM)   │
                                        │                         │
                                        │ • Rewrite Query         │
                                        │ • Expand Intent         │
                                        │ • Normalize Query       │
                                        └────────────┬────────────┘
                                                     │
                                                     ▼
                                        ┌─────────────────────────┐
                                        │ Semantic Cache (Redis)  │
                                        │                         │
                                        │ • Embedding Similarity  │
                                        │ • Cache Lookup          │
                                        └───────┬─────────┬───────┘
                                                │         │
                                        Cache Hit│         │Cache Miss
                                                │         ▼
                                                │
                                                │  ┌─────────────────────────┐
                                                │  │ Query Classification    │
                                                │  │                         │
                                                │  │ • Entity Lookup         │
                                                │  │ • Semantic Search       │
                                                │  │ • Counting / Ranking    │
                                                │  │ • Comparison            │
                                                │  │ • Temporal Queries      │
                                                │  │ • Multi-Hop Reasoning   │
                                                │  └────────────┬────────────┘
                                                │               │
                                                │               ▼
                                                │  ┌─────────────────────────────────────────┐
                                                │  │         Hybrid Retrieval Engine         │
                                                │  │                                         │
                                                │  │ • Dense Vector Search                  │
                                                │  │ • BM25 Keyword Search                  │
                                                │  │ • Neo4j Graph Retrieval                │
                                                │  └───────────────┬─────────────────────────┘
                                                │                  │
                                                │                  ▼
                                                │  ┌─────────────────────────────────────────┐
                                                │  │ Reciprocal Rank Fusion (RRF)           │
                                                │  └───────────────┬─────────────────────────┘
                                                │                  │
                                                │                  ▼
                                                │  ┌─────────────────────────────────────────┐
                                                │  │ Graph Boosting & Evidence Filtering     │
                                                │  └───────────────┬─────────────────────────┘
                                                │                  │
                                                │                  ▼
                                                │  ┌─────────────────────────────────────────┐
                                                │  │ Cross Encoder Re-Ranker                 │
                                                │  │ ms-marco-MiniLM-L-6-v2                  │
                                                │  └───────────────┬─────────────────────────┘
                                                │                  │
                                                │                  ▼
                                                │  ┌─────────────────────────────────────────┐
                                                │  │ Top Relevant Chunks + Graph Context     │
                                                │  └───────────────┬─────────────────────────┘
                                                │                  │
                                                │                  ▼
                                                │  ┌─────────────────────────────────────────┐
                                                │  │ Reasoning Agent                         │
                                                │  │                                         │
                                                │  │ • Aggregation                           │
                                                │  │ • Counting                              │
                                                │  │ • Ranking                               │
                                                │  │ • Comparisons                           │
                                                │  │ • Evidence Validation                   │
                                                │  │ • Sufficiency Checks                    │
                                                │  └───────────────┬─────────────────────────┘
                                                │                  │
                                                └──────────────────┤
                                                                   ▼
                                                ┌─────────────────────────────────────────┐
                                                │ NVIDIA Llama 3.3 70B                    │
                                                │ Grounded Answer Generation              │
                                                └───────────────┬─────────────────────────┘
                                                                │
                                                                ▼
                                                ┌─────────────────────────────────────────┐
                                                │ Final Response                          │
                                                │                                         │
                                                │ • Generated Answer                      │
                                                │ • Source Citations                      │
                                                │ • Confidence Score                      │
                                                │ • Query Classification                  │
                                                └─────────────────────────────────────────┘

=================================================================================
                               DOCUMENT INGESTION PIPELINE
=================================================================================


                    ┌───────────────────────────┐
                    │ PDF / Image / Screenshot  │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                    ┌───────────────────────────┐
                    │ PyMuPDF Text Extraction   │
                    └─────────────┬─────────────┘
                                  │
                          Enough Native Text?
                              ┌───┴───┐
                              │       │
                             Yes      No
                              │       │
                              ▼       ▼
                    ┌──────────────┐  ┌─────────────────────┐
                    │ Clean Text   │  │ OCR Pipeline        │
                    │ Processing   │  │ PaddleOCR           │
                    └──────┬───────┘  │ Tesseract Fallback  │
                           │          └──────────┬──────────┘
                           └──────────┬──────────┘
                                      │
                                      ▼
                    ┌───────────────────────────┐
                    │ Semantic Chunking         │
                    │ 512 Tokens + Overlap      │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                    ┌───────────────────────────┐
                    │ Metadata Enrichment       │
                    │                           │
                    │ • Page Number             │
                    │ • Section                 │
                    │ • OCR Confidence          │
                    │ • Extraction Method       │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                    ┌───────────────────────────┐
                    │ Entity & Relation         │
                    │ Extraction using LLM      │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                    ┌───────────────────────────┐
                    │ Entity Normalization      │
                    │ & Deduplication           │
                    └───────┬─────────┬─────────┘
                            │         │
                            ▼         ▼
                    ┌────────────┐ ┌────────────┐
                    │ Embeddings │ │ Neo4j      │
                    │ BGE Large  │ │ Knowledge  │
                    │            │ │ Graph      │
                    └─────┬──────┘ └─────┬──────┘
                          │              │
                          ▼              ▼
                    ┌────────────┐ ┌────────────┐
                    │ ChromaDB   │ │ Entity     │
                    │ Vector DB  │ │ Relations  │
                    └─────┬──────┘ └─────┬──────┘
                          │              │
                          └──────┬───────┘
                                 ▼
                    ┌───────────────────────────┐
                    │ BM25 Index Construction   │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                    ┌───────────────────────────┐
                    │ Hybrid Retrieval Ready    │
                    │ Knowledge Corpus          │
                    └───────────────────────────┘

## What This Project Uses

| **Area**                      | **Technology**                                                                                                                                                                                              |
| ----------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Backend API**               | FastAPI, Pydantic                                                                                                                                                                                           |
| **Frontend**                  | Next.js 14, Next.js                                                                                                                                                                                         |
| **LLM Provider**              | NVIDIA OpenAI-compatible API                                                                                                                                                                                |
| **Default LLM**               | `meta/llama-3.3-70b-instruct`                                                                                                                                                                               |
| **Embedding Model**           | NVIDIA Embedding API (`nvidia/llama-3.2-nv-embedqa-1b-v2`)                                                                                                                                                  |
| **Query Optimization**        | LLM-based Query Rewriting & Query Classification                                                                                                                                                            |
| **Chunking**                  | Semantic Chunking                                                                                                                                                                                           |
| **Reranker**                  | `cross-encoder/ms-marco-MiniLM-L-6-v2`                                                                                                                                                                      |
| **Vector Database**           | Qdrant Cloud (Cosine Similarity)                                                                                                                                                                            |
| **Graph Database**            | Neo4j AuraDB (Cloud)                                                                                                                                                                                        |
| **Keyword Search**            | BM25 (`rank-bm25`)                                                                                                                                                                                          |
| **Hybrid Retrieval**          | Dense Retrieval + BM25 + Knowledge Graph Retrieval + Reciprocal Rank Fusion (RRF)                                                                                                                           |
| **Reasoning**                 | Agentic Reasoning (Aggregation, Ranking, Counting, Comparison, Multi-hop Reasoning, Evidence Validation)                                                                                                    |
| **OCR & Document Processing** | Native PDF Extraction (PyMuPDF), PaddleOCR Fallback, Tesseract Fallback                                                                                                                                     |
| **Caching**                   | Semantic Cache (Redis) + Redis Chat Memory + In-Memory/Disk Fallback                                                                                                                                        |
| **Evaluation**                | Auto-generated verified QA pairs, Retrieval Evaluation (Recall@1/@3/@5, Precision@1/@3/@5, Hit Rate, MRR), Generation Evaluation (Faithfulness, Answer Relevancy, BERTScore), End-to-End Latency Evaluation |
| **Deployment**                | Docker, Docker Compose, Railway (Backend), Vercel (Frontend), Qdrant Cloud, Neo4j AuraDB                                                                                                                    |


## Core Features

- Upload PDFs and image documents through the API/frontend.
- Extract text from native PDFs and OCR scanned pages/images using PyMuPDF with PaddleOCR/Tesseract fallback.
- Perform semantic chunking while preserving document structure and metadata (page, chunk, section, extraction method, OCR confidence).
- Generate document summaries for improved document understanding.
- Extract financial/business entities and relationships into Neo4j AuraDB Knowledge Graph.
- Generate NVIDIA embeddings and store vectors in Qdrant Cloud.
- Retrieve documents using hybrid retrieval (Dense Vector Search + BM25 + Knowledge Graph Retrieval).
- Rewrite and optimize user queries using an LLM-based Query Rewriter before retrieval.
- Classify queries into entity lookup, semantic search, structured reasoning, temporal, relationship, multi-hop, aggregation, counting, ranking, comparison, and analytical query types.
- Fuse retrieval results using Reciprocal Rank Fusion (RRF), apply graph-based boosting, and rerank using a Cross-Encoder (ms-marco-MiniLM-L-6-v2).
- Execute an Agentic Reasoning module for aggregation, counting, ranking, comparison, multi-hop reasoning, evidence validation, and evidence sufficiency checks before answer generation.
- Generate grounded answers using NVIDIA Llama 3.3 70B with source citations and confidence scores.
- Accelerate repeated and semantically similar queries using Redis-based Semantic Caching and Redis-backed conversation memory.
- Evaluate retrieval quality using Recall@1/@3/@5, Precision@1/@3/@5, Hit Rate, and Mean Reciprocal Rank (MRR).
- Evaluate answer generation quality using Faithfulness, Answer Relevancy, and BERTScore (Precision, Recall, F1).
- Measure end-to-end pipeline latency, including Query Rewriting, Query Classification, Retrieval, RRF Fusion, Graph Boosting, Cross-Encoder Reranking, Evidence Validation, Reasoning, Semantic Cache Lookup, and LLM Generation.
- Deploy the production system using Railway (Backend), Vercel (Frontend), Qdrant Cloud, Neo4j AuraDB, Redis, and Docker.


## Query Classification

The classifier agent is lightweight and runs without an LLM call. It maps a query to a retrieval strategy and `top_k` depth.

Important query families include:

- Lookup: `ENTITY_LOOKUP`, `SEMANTIC_LOOKUP`, `LOOKUP`
- Structured reasoning: `NUMERICAL_FILTER`, `COUNTING`, `AGGREGATION`, `RANKING`
- Relationship reasoning: `COMPARISON`, `RELATIONSHIP`, `MULTI_HOP`, `TEMPORAL`, `ANALYTICAL`

These classifications control whether the system leans more on exact metadata, dense search, graph retrieval, wider corpus retrieval, or iterative retrieval.

## Main API Endpoints

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Check server health and indexed document count |
| `POST` | `/ingest` | Upload and ingest a PDF/image document |
| `POST` | `/query` | Ask a question over uploaded documents |
| `GET` | `/documents` | List indexed documents |
| `GET` | `/documents/{document_name}` | Get document metadata and summary |
| `DELETE` | `/documents/{document_name}` | Remove one document from indexes and disk |
| `POST` | `/reset` | Remove all documents |
| `GET` | `/entities` | Search graph entities |
| `GET` | `/entities/{entity_id}/neighbors` | Inspect graph neighbors |
| `POST` | `/evaluate` | Auto-generate verified QA pairs and evaluate |
| `POST` | `/evaluate/predefined` | Run predefined evaluation pairs |
| `GET` | `/cache/status` | Check cache readiness |
| `POST` | `/cache/clear` | Clear answer cache |
| `POST` | `/chat/clear` | Clear conversational memory |
| `POST` | `/debug/query` | Inspect classification and retrieval results |

## Environment Variables

Create `backend/.env` from `backend/.env.example`.

```env
NVIDIA_API_KEY=nvapi-------
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_MODEL=meta/llama-3.3-70b-instruct
NVIDIA_EMBEDDING_MODEL=nvidia/nv-embed-v1
NVIDIA_MAX_TOKENS=1024
NVIDIA_TEMPERATURE=0.1

GRAPH_BACKEND=neo4j

VECTOR_DB=qdrant
QDRANT_URL=https://your-qdrant-cluster-url.cloud.qdrant.io
QDRANT_API_KEY=your-api-key
QDRANT_COLLECTION=agentic-rag
NEO4J_URI=neo4j+s://52f6fec6.databases.neo4j.io
NEO4J_USERNAME=52f6fec6
NEO4J_PASSWORD=7XcwEdmn1upF-Is02B7GiUgLVeq4lltV8Me33TTjkUc
NEO4J_DATABASE=52f6fec6
AURA_INSTANCEID=52f6fec6
AURA_INSTANCENAME=Free instance


# Agentic RAG settings
QUERY_AGENT_ENABLED=true
REASONING_AGENT_ENABLED=true
MAX_RETRIEVAL_ITERATIONS=3
CACHE_LOAD_MONITORING=true
CACHE_READY_PERCENT=100

#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
```

Qdrant Cloud setup:
Create a Qdrant Cloud cluster and get your URL and API Key. The app will automatically create the `agentic-rag` collection configured for 4096 dimensions with Cosine distance and BM25 sparse vectors enabled.

Neo4j is required because `GRAPH_BACKEND` is configured for Neo4j only. Redis is optional; if Redis is unavailable, the app falls back to in-memory and local disk caching where supported.

```

## Docker

```bash
docker-compose up --build
```

The compose file starts the FastAPI backend and Next.js frontend. Ensure required external services such as Neo4j, and optionally Redis, are available or configured for your environment.

## Typical Usage

1. Start Neo4j.
2. Configure `backend/.env` with your NVIDIA API key and model settings.
3. Start the backend and frontend.
4. Upload a PDF, scanned PDF, screenshot, or image.
5. Wait for ingestion to complete: OCR/text extraction, chunking, entity abstraction, embeddings, Qdrant/ChromaDB insert, Neo4j graph insert, and BM25 rebuild.
6. Ask questions from the frontend or `/query`.
7. Review answer sources, confidence, and query classification.
8. Run `/evaluate` to measure retrieval and grounding quality on the current corpus.

## Evaluation

The project includes an automated end-to-end evaluation pipeline. It samples indexed chunks, prompts the configured LLM to generate factual, relational, comparative, and reasoning-based questions, verifies each question against the source documents, and evaluates both retrieval performance and answer generation quality.

### Retrieval Evaluation
- Recall@1, Recall@3, Recall@5
- Precision@1, Precision@3, Precision@5
- Hit Rate@1, Hit Rate@3, Hit Rate@5
- Mean Reciprocal Rank (MRR)

### Generation Evaluation
- Faithfulness
- Answer Relevancy
- BERTScore (Precision, Recall, F1)

### Pipeline Latency Evaluation
- Query Rewriter Latency
- Query Classification Latency
- Dense Retrieval Latency
- BM25 Retrieval Latency
- Graph Retrieval Latency
- Hybrid Retrieval Latency
- Reciprocal Rank Fusion (RRF) Latency
- Graph Boosting Latency
- Cross-Encoder Reranking Latency
- Evidence Validation Latency
- Reasoning Agent Latency
- Semantic Cache Lookup Latency
- LLM Generation Latency
- End-to-End Response Latency

The evaluation dashboard provides both **per-question metrics** and **overall aggregate metrics**, enabling comprehensive analysis of retrieval quality, answer generation quality, and system performance across the complete Agentic RAG pipeline.

## Notes

- The LLM is used for query rewriting, grounded answer generation, document summarization, entity and relationship extraction, reasoning, and automatic evaluation question generation.
- NVIDIA Embedding API is used to generate high-dimensional semantic embeddings for both documents and queries.
- Qdrant Cloud persists document embeddings and performs dense vector similarity search.
- Neo4j AuraDB stores entities, document mentions, and semantic relationships extracted from document chunks for knowledge graph retrieval.
- The retrieval pipeline combines Dense Vector Search, BM25 Keyword Search, and Knowledge Graph Retrieval using Reciprocal Rank Fusion (RRF), followed by graph boosting and Cross-Encoder reranking.
- A reasoning agent performs aggregation, counting, ranking, comparison, multi-hop reasoning, and evidence sufficiency validation before answer generation.
- Redis-based semantic caching accelerates repeated and semantically similar queries while Redis-backed chat memory maintains conversational context.
- The final response is generated only after query rewriting, hybrid retrieval, reranking, reasoning, evidence validation, and confidence estimation, with source citations for grounded answers.
- The system includes an automated evaluation framework measuring retrieval quality (Recall@K, Precision@K, Hit Rate, MRR), generation quality (Faithfulness, Answer Relevancy, BERTScore), and end-to-end pipeline latency.
