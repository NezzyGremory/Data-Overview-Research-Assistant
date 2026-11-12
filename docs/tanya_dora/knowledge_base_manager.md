# Knowledge Base Manager

## Studio features

**Tanya DORA Studio → Knowledge Base** uses `DoraKnowledgeBase`, the same local store consumed by the RAG pipeline. It supports PDF, DOCX, TXT, Markdown, RST, and HTML import; document list and status; title, author, year, DOI, URL, license, category, and verification metadata; keyword search; indexed-text preview; metadata update; re-index; and delete. Identical source bytes are rejected with the already-indexed document ID so a duplicate does not create a second index entry.

The current backend stores source copies below `get_data_dir()/tanya_dora/knowledge/sources/`, metadata/chunks in SQLite, and uses FTS5 keyword retrieval. Ollama embeddings can be selected for optional hybrid retrieval when a compatible local embedding model is installed. Embedding generation is not part of the basic keyword index.

## Safety and source handling

Import preserves the original local file and records its SHA-256. Default verification is `UNVERIFIED`; indexing does not certify a source. PDF extraction rejects encrypted/oversized files and preserves page numbers. DOCX extraction validates the archive size and entry count. Scanned PDFs need OCR, which is not provided. Search citations are composed from stored metadata and page/section values; missing values are not invented.

## Validation status

Automated tests cover text/HTML import, chunking, FTS5 search, metadata, re-index, deletion, limits, duplicate source rejection, and synthetic-vector hybrid retrieval. A manual local integration check imported a generated two-page PDF containing Indonesian and English text, made three chunks, retrieved both language queries with `Page 1`/`Page 2` citations, rejected duplicate bytes, returned no results for an unrelated query, and rejected both an empty-text and corrupt PDF. This checks PDF parsing and citation metadata, not an independently sourced research paper or answer quality. It ran under global Python 3.12 with PyMuPDF; the project `.venv` lacks PyMuPDF and `python-docx`, so PDF/DOCX operations in the application environment remain blocked until `requirements_ai.txt` is installed. Ollama was unavailable, so live embeddings, semantic/hybrid retrieval, and model-grounded RAG were not verified.
