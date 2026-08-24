# Atman Cloud RAG System — Comprehensive Architecture, Progress & Decision Log

> **Current Status:** Phases 0, 1, 2, and 3 are **100% Complete & Verified** (44.4% of total project).  
> **Test Status:** 14/14 automated tests passing in ~26 seconds across Ingestion, Chunking, and Vector Store.

---

## 1. Executive Summary & System Objectives

The goal of this project is to build a production-grade, local Retrieval-Augmented Generation (RAG) Document Q&A system over **7 heterogeneous enterprise PDF documents** provided by Atman Cloud Consultants:
1. `Product_Manual.pdf` (Structured technical manual, troubleshooting tables)
2. `Employee_Handbook.pdf` (HR/company policies, leave, benefits)
3. `API_Reference.pdf` (Technical specifications, endpoints, code snippets)
4. `FAQ_Support.pdf` (Question-and-answer pairs, support triage)
5. `Security_Policy.pdf` (Compliance, encryption, defined terms)
6. `Pricing_and_SLA.pdf` (Tier pricing matrices, SLA percentage tables)
7. `Onboarding_Guide.pdf` (Day-by-day chronological workflow guides)

### Core Architectural Mandates
* **100% Local Dense Embeddings & Storage:** Zero external network dependency or API costs for document indexing and vector retrieval.
* **Preservation of Tabular Structure:** Tables are extracted as 2D cell matrices and serialized to Markdown pipe tables; they are never fragmented.
* **Type-Aware Semantic Chunking:** Different documents require different splitting strategies (Q&A boundaries, code/endpoints, narrative paragraphs, atomic tables).
* **Strict Source Provenance & Metadata:** Every chunk carries document codes (e.g. `PRC-SLA-021`), version (`3.2`), page number, section heading, and chunk type.
* **Idempotent Ingestion & Fast Startup:** Sub-second application boot times by verifying existing vector counts before triggering re-embedding.

---

## 2. End-to-End File Map & Architecture

### System Interconnection Diagram

```mermaid
graph TD
    subgraph Config Layer
        CFG[config/config.py<br/>Global Settings & Paths]
        LOG[config/logging_config.py<br/>Structured Logger]
    end

    subgraph Phase 1: Ingestion Engine
        PDF[files/*.pdf<br/>7 Heterogeneous PDFs] --> EXT[src/ingestion/pdf_extractor.py<br/>PDFExtractor]
        EXT -->|Extracts 2D Grid| TAB[src/ingestion/table_processor.py<br/>TableProcessor]
        EXT -->|Emits Schema| MOD1[src/ingestion/models.py<br/>PageDocument & ExtractedTable]
    end

    subgraph Phase 2: Semantic Chunking Engine
        MOD1 -->|List of PageDocuments| CHK[src/chunking/chunker.py<br/>SemanticChunker Router]
        CHK -->|Route by doc_type| SPL[src/chunking/splitters.py<br/>4 Specialized Splitters]
        SPL -->|Text / Table / Code / QA| ENR[src/chunking/enricher.py<br/>MetadataEnricher]
        ENR -->|Enriched Chunks| MOD2[src/chunking/models.py<br/>EnrichedChunk Pydantic Model]
    end

    subgraph Phase 3: Embedding & Vector Storage
        MOD2 -->|47 EnrichedChunks| IDX[src/vector_store/indexer.py<br/>CorpusIndexer Pipeline]
        IDX -->|Batch Text| EMB[src/vector_store/embedder.py<br/>EmbeddingEngine (all-MiniLM-L6-v2)]
        EMB -->|384-dim Vectors| CHR[src/vector_store/chroma_store.py<br/>ChromaVectorStore (PersistentClient)]
        CHR -->|Store HNSW Index| DB[(chroma_db/<br/>Local SQLite/HNSW)]
    end

    CFG -.-> EXT
    CFG -.-> SPL
    CFG -.-> EMB
    CFG -.-> CHR
    LOG -.-> EXT
    LOG -.-> CHK
    LOG -.-> IDX
```

---

## 3. Detailed File-by-File Breakdown & Connections

Below is the exhaustive list of all 14 project files created across Phases 0–3, how they link together, and their functional contracts.

```
d:\Codes\Assignment\
├── config/
│   ├── __init__.py               # Re-exports config and logger
│   ├── config.py                 # Central environment & hyperparameter configuration
│   └── logging_config.py         # Colorized, formatted console logging
├── src/
│   ├── ingestion/
│   │   ├── __init__.py           # Re-exports PDFExtractor, TableProcessor, Models
│   │   ├── models.py             # Pydantic schemas: ExtractedTable, PageDocument
│   │   ├── table_processor.py    # Cell cleaning, whitespace normalization, Markdown pipe table generator
│   │   ├── pdf_extractor.py      # Dual-engine extractor: pdfplumber (primary) + PyMuPDF (fallback)
│   │   └── verify_extraction.py  # Smoke test verifying all 7 PDFs, 20 pages, 11 tables
│   ├── chunking/
│   │   ├── __init__.py           # Re-exports SemanticChunker, splitters, models
│   │   ├── models.py             # EnrichedChunk schema, DOC_TYPE_MAP, Chroma metadata flattener
│   │   ├── enricher.py           # MetadataEnricher: regex heading detection, token estimation
│   │   ├── splitters.py          # 4 splitters: TextSplitter, AtomicTableChunker, CodeBlockChunker, QABoundarySplitter
│   │   └── chunker.py            # SemanticChunker: orchestrates doc-type routing & table atomicity
│   └── vector_store/
│       ├── __init__.py           # Re-exports EmbeddingEngine, ChromaVectorStore, CorpusIndexer
│       ├── embedder.py           # SentenceTransformer wrapper (all-MiniLM-L6-v2, 384 dimensions)
│       ├── chroma_store.py       # ChromaDB PersistentClient with cosine distance and scalar metadata
│       └── indexer.py            # Automated end-to-end indexer with idempotency check and CLI --force flag
└── tests/
    ├── conftest.py               # Shared pytest configuration & paths
    ├── test_ingestion.py         # 4 tests: PDF extraction, table matrix parsing, metadata extraction
    ├── test_chunking.py          # 5 tests: chunk counts, table atomicity, Q&A pairs, metadata schema
    └── test_vector_store.py      # 5 tests: 384-dim check, upsert count, semantic recall, filtering, idempotency
```

### Deep Connection Matrix

| File Path | Role & Exports | Upstream Dependencies | Downstream Consumers |
|---|---|---|---|
| `config/config.py` | Global settings (`DOCS_DIR`, `CHROMA_DB_DIR`, chunk sizes, model names) | Environment variables, `.env` | All modules (`pdf_extractor`, `splitters`, `embedder`, `chroma_store`) |
| `src/ingestion/models.py` | `ExtractedTable`, `PageDocument` Pydantic models | Pydantic | `pdf_extractor.py`, `table_processor.py`, `chunker.py`, `splitters.py` |
| `src/ingestion/table_processor.py` | `TableProcessor.clean_cell()`, `to_markdown()` | `ExtractedTable` | `pdf_extractor.py` (during PDF page extraction) |
| `src/ingestion/pdf_extractor.py` | `PDFExtractor.extract_document()`, `extract_all()` | `pdfplumber`, `pymupdf`, `TableProcessor`, `PageDocument` | `verify_extraction.py`, `chunker.py`, `indexer.py`, `test_ingestion.py` |
| `src/chunking/models.py` | `EnrichedChunk`, `DOC_TYPE_MAP`, `.to_chroma_metadata()` | Pydantic | `enricher.py`, `splitters.py`, `chunker.py`, `chroma_store.py`, `test_chunking.py` |
| `src/chunking/enricher.py` | `MetadataEnricher.detect_heading()`, `estimate_tokens()`, `enrich()` | `EnrichedChunk` | `splitters.py`, `chunker.py` |
| `src/chunking/splitters.py` | `TextSplitter`, `AtomicTableChunker`, `CodeBlockChunker`, `QABoundarySplitter` | `cfg`, `ExtractedTable`, `MetadataEnricher`, `EnrichedChunk` | `chunker.py`, `test_chunking.py` |
| `src/chunking/chunker.py` | `SemanticChunker.chunk_page()`, `chunk_all()` | `PageDocument`, `DOC_TYPE_MAP`, 4 splitters from `splitters.py` | `indexer.py`, `test_chunking.py` |
| `src/vector_store/embedder.py` | `EmbeddingEngine.embed_texts()`, `embed_query()`, `.dimension` | `sentence_transformers`, `cfg.EMBEDDING_MODEL` | `indexer.py`, `test_vector_store.py`, `src/retrieval/` (Phase 4) |
| `src/vector_store/chroma_store.py` | `ChromaVectorStore.upsert_chunks()`, `query()`, `count()`, `reset_collection()` | `chromadb`, `cfg.CHROMA_DB_DIR`, `EnrichedChunk` | `indexer.py`, `test_vector_store.py`, `src/retrieval/` (Phase 4) |
| `src/vector_store/indexer.py` | `CorpusIndexer.index_corpus()` + CLI runner | `PDFExtractor`, `SemanticChunker`, `EmbeddingEngine`, `ChromaVectorStore` | CLI execution (`python -m src.vector_store.indexer`), Streamlit startup |

---

## 4. Key Decisions, Rationale & Direct Code Impacts

This section answers: **"Why did we make this decision, what were the alternatives, and what code did it produce?"**

---

### Decision 1: Table Extraction Engine (`pdfplumber` over `PyMuPDF`/`pypdf`)
* **The Situation:** Enterprise PDFs contain multi-column pricing and SLA guarantee tables (e.g. `Pricing_and_SLA.pdf` Page 2).
* **The Decision:** Use `pdfplumber.extract_tables()` as the primary engine with cell-by-cell matrix extraction.
* **Why not PyMuPDF or pypdf alone?** PyMuPDF and pypdf extract text as flat sequential strings. They strip 2D spatial alignment, merging columns into unparseable text blobs (`"Free Standard Enterprise $0 $12 Custom 5GB 500GB Unlimited"`).
* **Impact on Code:**
  * Created `TableProcessor` (`src/ingestion/table_processor.py`) which converts raw cell grids `List[List[Optional[str]]]` into clean Markdown pipe tables with aligned header rows and separator lines (`|---|---|`).
  * Yielded 100% extraction accuracy across all 11 tables in the 7 PDFs with 0 fallback triggers.

---

### Decision 2: Atomic Table Preservation & Strategy Routing (`DOC_TYPE_MAP`)
* **The Situation:** Heterogeneous documents require different segmentation strategies. If a standard sliding character window (e.g. 800 chars) cuts through a Markdown table, the table becomes unusable.
* **The Decision:**
  1. **Atomic Tables:** `AtomicTableChunker` treats each `ExtractedTable` as an indivisible chunk, prepending contextual provenance metadata (`[DOCUMENT: Pricing_and_SLA | DOC_CODE: PRC-SLA-021 | PAGE: 2]`).
  2. **Strategy Routing (`DOC_TYPE_MAP`):**
     * `FAQ_Support` $\rightarrow$ `QABoundarySplitter` (splits on `Q:` boundaries so questions and answers stay together).
     * `API_Reference` $\rightarrow$ `CodeBlockChunker` (splits along `POST /api/...`, `GET /api/...` HTTP endpoints).
     * Policies/Manuals $\rightarrow$ `TextSplitter` (recursive character splitting: paragraph $\rightarrow$ newline $\rightarrow$ sentence $\rightarrow$ space).
* **Impact on Code:**
  * Created `src/chunking/splitters.py` (4 distinct splitters).
  * Created `SemanticChunker.chunk_page()` (`src/chunking/chunker.py`) which processes tables first, then routes remaining text through the type-specific splitter.
  * Verified in `test_pricing_sla_tables_atomic`: all 3 tiers (`Free`, `Standard`, `Enterprise`) and `99.5%` SLA are preserved in complete, unfragmented table chunks.

---

### Decision 3: Metadata Flattening for Vector Storage (`to_chroma_metadata`)
* **The Situation:** ChromaDB stores vector embeddings alongside metadata dictionaries, but its underlying SQLite schema strictly rejects nested dictionaries or lists.
* **The Decision:** Explicitly define `EnrichedChunk.to_chroma_metadata()` in `src/chunking/models.py` which extracts only primitive scalar values (`str`, `int`, `float`, `bool`).
* **Impact on Code:**
  * Every chunk stored in ChromaDB contains: `chunk_id`, `doc_name`, `doc_filename`, `doc_code`, `doc_version`, `page_num`, `section_heading`, `chunk_type`, `char_count`, `token_count_est`, `has_table`, `has_code`, `doc_type`, `ingestion_ts`.
  * Enables fast retrieval pre-filtering (`where={"doc_name": "FAQ_Support"}`).

---

### Decision 4: Local Sentence-Transformers Embedding vs. OpenAI API
* **The Situation:** Need dense semantic embeddings to represent chunk texts and search queries.
* **The Decision:** Standardize on `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors).
* **Why not OpenAI `text-embedding-3-small`?**
  1. Local model ensures 100% data confidentiality and zero external API dependencies for evaluation.
  2. Zero latency from network roundtrips: entire corpus (47 chunks) embeds in <2 seconds on CPU.
  3. Evaluators can clone and run `uv sync` without needing paid API credentials.
* **Impact on Code:**
  * Created `EmbeddingEngine` (`src/vector_store/embedder.py`) with batch inference and query embedding methods.

---

### Decision 5: Length-Invariant Cosine Distance Space
* **The Situation:** Chunk lengths vary widely—from a 120-character short FAQ answer to an 800-character handbook policy paragraph.
* **The Decision:** Configure ChromaDB collection with `metadata={"hnsw:space": "cosine"}`.
* **Why not Euclidean ($L_2$) or Dot Product?** $L_2$ distance scales with vector magnitude (and text length), unjustly penalizing short, precise answers. Cosine similarity evaluates angle rather than magnitude, ensuring fair matching across variable chunk sizes.
* **Impact on Code:**
  * Configured in `ChromaVectorStore.__init__()` (`src/vector_store/chroma_store.py`).
  * Similarity formula established: $\text{similarity} = 1 - \text{cosine\_distance}$.

---

### Decision 6: Corpus Idempotency & CLI Re-indexing
* **The Situation:** When the Streamlit UI or CLI starts up, re-extracting, re-chunking, and re-embedding 7 PDFs on every launch would add unnecessary 3-second delays.
* **The Decision:** Implement an idempotency check in `CorpusIndexer.index_corpus()`:
  * If `vector_store.count() > 0` and `force_reindex=False`, return immediately (`skipped=True`).
  * If `--force` is passed, wipe collection via `reset_collection()` and rebuild from scratch.
* **Impact on Code:**
  * `CorpusIndexer` (`src/vector_store/indexer.py`) serves as the single entrypoint for database indexing, providing instant sub-second app startup.

---

## 5. Concrete Data Flow Walkthrough

Here is what happens to a single real page (`Pricing_and_SLA.pdf`, Page 2) as it travels through the completed pipeline:

```
[Raw PDF Bytes on Disk]
       │
       ▼ (PDFExtractor via pdfplumber)
PageDocument:
  doc_name: "Pricing_and_SLA"
  page_num: 2
  doc_code: "PRC-SLA-021"
  doc_version: "3.2"
  has_tables: True
  tables: [
     ExtractedTable(
        table_id: "Pricing_and_SLA__p002__t01",
        markdown_content: "| Tier | Monthly Price | Storage Included | Users |\n|---|---|---|---|\n| Free | $0 | 5 GB | 1 |\n| Standard | $12/user/month | 500 GB pooled | Up to 25 |\n| Enterprise | Custom | Unlimited | Unlimited |",
        row_count: 4, col_count: 4
     ),
     ExtractedTable(
        table_id: "Pricing_and_SLA__p002__t02",
        markdown_content: "| Tier | Uptime Guarantee | Support SLA |\n|---|---|---|\n| Standard | 99.5% monthly | Next business day, email |\n| Enterprise | 99.95% monthly | 1 hour for Sev-1, 24/7 phone+email |",
        row_count: 3, col_count: 3
     )
  ]
       │
       ▼ (SemanticChunker & AtomicTableChunker)
EnrichedChunk:
  chunk_id: "Pricing_and_SLA__p002__t01"
  chunk_type: "table"
  has_table: True
  section_heading: "Table: Tier, Monthly Price, Storage Included"
  text: "[DOCUMENT: Pricing_and_SLA | DOC_CODE: PRC-SLA-021 | PAGE: 2 | TABLE: Pricing_and_SLA__p002__t01]\n| Tier | Monthly Price | Storage Included | Users |\n|---|---|---|---|\n| Free | $0 | 5 GB | 1 |\n| Standard | $12/user/month | 500 GB pooled | Up to 25 |\n| Enterprise | Custom | Unlimited | Unlimited |"
  char_count: 312
  token_count_est: 78
       │
       ▼ (EmbeddingEngine via all-MiniLM-L6-v2)
Vector Embedding:
  [0.0234, -0.0512, 0.0891, ..., 0.0118] (384 floats)
       │
       ▼ (ChromaVectorStore via PersistentClient)
ChromaDB HNSW / SQLite Index:
  ID: "Pricing_and_SLA__p002__t01"
  Document: text string
  Embedding: [384 floats]
  Metadata: {"doc_name": "Pricing_and_SLA", "doc_code": "PRC-SLA-021", "page_num": 2, "chunk_type": "table", ...}
```

---

## 6. Empirical Benchmarks & Verification Summary

### Document Corpus Metrics

| Document Name | Total Pages | Extracted Tables | Doc Code | Version | Output Chunks | Primary Doc Type |
|---|:---:|:---:|:---:|:---:|:---:|---|
| `API_Reference.pdf` | 3 | 2 | `API-REF-002` | 3.2 | 7 | `technical_api` |
| `Employee_Handbook.pdf` | 3 | 0 | `HR-EH-2026` | 3.2 | 7 | `narrative_policy` |
| `FAQ_Support.pdf` | 2 | 0 | `FAQ-SUP-014` | 3.2 | 9 | `qa_pairs` |
| `Onboarding_Guide.pdf` | 4 | 2 | `ONB-GDE-009` | 3.2 | 7 | `chronological_guide` |
| `Pricing_and_SLA.pdf` | 2 | 2 | `PRC-SLA-021` | 3.2 | 5 | `data_tables` |
| `Product_Manual.pdf` | 3 | 2 | `PM-CSP-001` | 3.2 | 6 | `structured_manual` |
| `Security_Policy.pdf` | 3 | 1 | `SEC-POL-007` | 3.2 | 6 | `policy_defined_terms` |
| **TOTALS** | **20** | **11** | — | — | **47** | — |

### Execution Performance
* **PDF Extraction Time:** ~0.65 seconds (all 7 PDFs, 20 pages).
* **Chunking Time:** ~0.08 seconds (20 pages $\rightarrow$ 47 enriched chunks).
* **Embedding & Indexing Time:** **2.83 seconds** (47 vectors generated and persisted to disk).
* **Automated Test Suite:** **14/14 passed in 26.67 seconds** (`pytest tests/`).

---

## 7. Roadmap to Completion (Phases 4–8)

With the foundational data layer complete, here is how the remaining phases connect into this architecture:

```
                  ┌──────────────────────────────────────────────────────────┐
                  │                 CURRENTLY COMPLETED                      │
                  │   Phase 0 (Setup) ──► Phase 1 (Ingest) ──► Phase 2 (Chunk) │
                  │                                  │                       │
                  │                                  ▼                       │
                  │                           Phase 3 (Index)                │
                  └──────────────────────────────────┬───────────────────────┘
                                                     │
                                                     ▼
                  ┌──────────────────────────────────────────────────────────┐
                  │                    UPCOMING PHASES                       │
                  │                                                          │
                  │  Phase 4: Two-Stage Retrieval                            │
                  │           ├── Stage 1: Dense ChromaDB query (top-10)     │
                  │           ├── Stage 2: Cross-Encoder reranker (top-3)    │
                  │           └── Guardrail Layer 1: Abstention (<0.40 score)│
                  │                                  │                       │
                  │                                  ▼                       │
                  │  Phase 5: Multi-Provider LLM Generation                  │
                  │           ├── Groq Cloud (llama-3.3-70b) primary         │
                  │           ├── Ollama (llama3.2) offline fallback         │
                  │           ├── Grounded prompt + strict provenance        │
                  │           └── Guardrail Layer 2: Citation validator      │
                  │                                  │                       │
                  │                                  ▼                       │
                  │  Phase 6: Streamlit Interactive UI                       │
                  │           ├── Question input, answer streaming           │
                  │           ├── Confidence score badges (High/Med/Low)     │
                  │           └── Expandable source provenance inspector     │
                  │                                  │                       │
                  │                                  ▼                       │
                  │  Phase 7: Empirical Evaluation Suite                     │
                  │           ├── 15 benchmark questions (in-domain & OOD)   │
                  │           └── Automated metrics -> eval_log.json         │
                  │                                  │                       │
                  │                                  ▼                       │
                  │  Phase 8: README, System Manual & Final Packaging        │
                  └──────────────────────────────────────────────────────────┘
```
