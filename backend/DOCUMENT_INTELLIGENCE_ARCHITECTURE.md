# Document Intelligence Extension

This backend now extends the existing RAG stack without replacing the parts that
already work: ChromaDB dense retrieval, BM25, RRF, cross-encoder reranking,
summarization, cache, memory, hallucination controls, and evaluation remain in
place.

## Runtime Flow

```text
Upload PDF/Image
  -> native PyMuPDF extraction
  -> OCR fallback for scanned/low-text pages
       PaddleOCR when installed, Tesseract fallback
  -> OCR/text cleaning
  -> structure-aware chunking
  -> entity extraction + normalization + dedup
  -> ChromaDB chunk/vector upsert
  -> Neo4j graph upsert

Query
  -> query entity extraction
  -> graph entity lookup + chunk expansion
  -> BM25 retrieval
  -> dense Chroma retrieval
  -> RRF merge + graph boost
  -> cross-encoder rerank
  -> grounded generation with citations/confidence
```

## New Modules

- `app/ocr.py`: page-level native extraction plus PaddleOCR/Tesseract fallback.
- `app/entities.py`: deterministic entity extraction, normalization, dedup hooks,
  and optional spaCy augmentation.
- `app/graph.py`: Neo4j-backed graph storage and graph retrieval helpers.
- `app/ingestion.py`: OCR-aware parsing, text cleanup, section-aware chunking.
- `app/retrieval.py`: BM25 + dense + graph retrieval orchestration.

## Chunk Metadata

Chunks now carry:

```json
{
  "document": "contract.pdf",
  "page": 1,
  "chunk_index": 0,
  "section_title": "Section 2. Parties",
  "entities": ["person:...", "org:..."],
  "ocr_confidence": 0.92,
  "extraction_method": "native|paddleocr|tesseract"
}
```

## Graph Schema

Entity nodes:

```text
(:Entity {
  entity_id,
  label,
  name,
  aliases,
  confidence,
  metadata
})
```

Document links and inferred relationships:

```text
(:Document)-[:MENTIONS {page, chunk_index, text}]->(:Entity)
(:Entity)-[:RELATED {type: "CO_OCCURS_WITH", confidence, evidence}]->(:Entity)
```

Neo4j stores document nodes, entity nodes, mention edges, and inferred
relationships. The application expects `GRAPH_BACKEND=neo4j`.

## API Additions

- `POST /ingest`: now accepts PDFs and images.
- `GET /entities?q=<name>`: search indexed graph entities.
- `GET /entities/{entity_id}/neighbors?depth=1`: graph neighbor lookup.
- `POST /debug/query`: includes graph retrieval candidate count.

## Docker Notes

Neo4j starts with the default compose stack:

```bash
docker compose up --build
```

For OCR, the Dockerfile installs Tesseract system packages. PaddleOCR is in
`requirements.txt`; for GPU PaddleOCR, pin the matching `paddlepaddle-gpu`
package for your CUDA runtime.
