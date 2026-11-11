# Tanya DORA Completion Audit

Audit date: 2026-11-12  
Scope: existing Tanya DORA services, their current UI wiring, local SQLite stores, documentation, dependencies, and regression tests. This report records the state before completion work.

## Status key

- **Implemented**: a concrete service or control exists in the source.
- **Integrated**: the service is called by the active Tanya DORA UI or another requested flow.
- **Tested**: covered by automated tests using the environment stated below.
- **Verified**: exercised against the real local runtime or a real document/model.
- **Not tested**: no execution evidence yet.
- **Blocked**: execution requires a local service, model, hardware, or optional dependency that is absent.

## Component audit

| Component | Implemented | Integrated | Tested / verified | Gaps found |
| --- | --- | --- | --- | --- |
| Ollama client and model registry (`model_registry.py`, `llm_manager.py`) | Loopback-only HTTP client; `/api/tags`, `/api/ps`, chat, stream, cancel, embeddings, timeout/error paths; refuses unknown/uninstalled models and never pulls automatically. | Tanya DORA chat model selector, stream, stop, status, RAG and tool loop. | HTTP behavior, stream cancellation, errors and model availability have mock tests. Live endpoint and inference are **blocked**. | No dedicated model test panel or response timer. No in-app install preflight/download consent flow. Ollama CLI and endpoint are unavailable here. |
| Knowledge Base (`knowledge_base.py`) | Local source copies, TXT/MD/RST/HTML/PDF/DOCX parsing, metadata, hash, chunks, FTS5, local embeddings, hybrid retrieval, re-index/delete/update. | RAG chat uses keyword or hybrid retrieval and exposes citations. | Service tests cover text/HTML, FTS, synthetic embeddings, metadata, delete and limits. No repository PDF/DOCX fixtures; PDF/DOCX parser packages are absent from `.venv`, so extraction is **not tested** in the project environment. No real Ollama embeddings. | No document-manager UI. Search quality and language coverage have not been evaluated on real documents. |
| RAG (`rag_pipeline.py`) | Bounded context, source filtering, citations from retrieved chunks, history/tool support, no-source refusal. | Tanya DORA chat calls the RAG service. | Mock LLM and synthetic result tests cover source handling, hybrid path, bounds, tools and no-source refusal. Model-grounded output is **not verified**. | No real-model citation faithfulness check. |
| Conversation history (`conversation_store.py`) | Separate local SQLite conversations/messages, source metadata, rename/delete/clear. | Tanya DORA chat. | Persistence and local CRUD tests. | No schema migration/version test. |
| Read-only DORA tools (`tool_calling.py`) | Bounded schemas and outputs; cached analyzer snapshots, DKB and opt-in OpenAlex/Crossref; no model code execution. | Registered from the active chat context. | Mock tool execution and tool-loop tests. Academic providers are not live-tested. | No UI to inspect tool configuration/results beyond chat activity/source panel. |
| TDID dataset (`tdid_dataset.py`) | JSONL/CSV import, validation, edits, exact/repeated-prompt groups, category/difficulty/verification counts, grouped splits, immutable hash-checked versions and export. | Fine-tuning engine consumes a saved split. | Unit tests cover import errors, validation, split leakage, verification and snapshot integrity. | No dataset manager UI; no curated verified trial dataset has been trained. |
| Fine-tuning (`fine_tuning.py`) | Local-only preflight, SFT + LoRA/QLoRA runner, resource estimates, checkpoint/progress callback, cancellation and manifest. | No UI caller yet. | Preflight/config tests only. Actual training is **blocked**: `.venv` has no torch, transformers, datasets, peft, trl, accelerate or bitsandbytes. | No run/checkpoint/loss/evaluation/reload evidence. No workspace export UI. |
| Adapter/model registry (`model_versioning.py`) | Adapter directory validation, SHA-256, import/export ZIP, structural base fingerprint, provenance and explicit registry selection. | Not connected to Tanya DORA inference. | Unit tests cover registry integrity, archive safety, compatibility fingerprint and selection confirmation. | **No adapter runtime integration**, model compatibility decision, activation/deactivation, base-model history or rollback in the active chat path. The base fingerprint is structural (config plus weight filenames/sizes), not a hash of every base weight. |
| TDEB (`evaluation_benchmark.py`) | Separate benchmark snapshots, TDID leakage gate, target callback/Ollama target, response/latency/error persistence, transparent lexical proxies and human-review storage. | No evaluation UI. | Deterministic callback tests only. No live model comparisons. | No UI for the five requested conditions. Automated proxy metrics do not establish factuality, reasoning, citation accuracy or hallucination rate. |
| Tanya DORA chat UI (`tanya_dora_page.py`, `main_window.py`) | Lazy page, local model selector/status, conversation history, stream/stop, RAG citations, tools and optional online research. | Routed in the active `MainWindow` page stack. | Qt tests and a 1440 x 900 screenshot (`screenshots/tanya_dora_page.png`). No live model inference. | No management studio or model-test panel; response duration is not shown. |

## Storage and migration review

- Existing project/research data remains in `app/storage/database.py`; its established additive column migrations are separate from Tanya DORA. No SINTA or project database implementation was changed for this audit.
- Tanya DORA stores conversations, knowledge, TDID versions, model versions/settings, and benchmark runs in separate SQLite files below `get_data_dir()/tanya_dora/`. TDID and benchmark snapshots are stored as files and checked by hashes.
- The new stores initialize tables with `CREATE TABLE IF NOT EXISTS` (and FTS5 for the knowledge index). They do not yet use `PRAGMA user_version`, explicit migration ledgers, or compatibility migrations for a prior Tanya DORA schema. Fresh-store creation and persistence are tested; upgrade-from-older-AI-store behavior is not.
- Runtime database contents were not inspected. The audit reviewed schemas and migration code only; no user data or `data/` files were modified.

## Documentation review

Reviewed the existing `docs/tanya_dora/` architecture, local setup, KB, RAG, dataset, training, hardware, model compatibility/versioning, tool calling, evaluation, and troubleshooting pages plus the example JSON/JSONL files. The documents correctly disclose several blockers, but they describe phases 1–9 as implemented while the management UI, actual adapter runtime, live validation, training run, and real benchmark remain outstanding. These distinctions must stay explicit in follow-up docs.

The project design references `design/html_export/` and `design/tokens.md` are absent in this checkout. Use the existing application theme and spacing conventions for the requested single Studio page.

## Runtime and dependency evidence

- `.venv` is Python 3.13.12 with PySide6 and matplotlib available.
- `.venv` does not contain PyMuPDF (`fitz`), python-docx (`docx`), torch, transformers, datasets, peft, trl, accelerate, or bitsandbytes.
- No `ollama` executable was found. `GET http://127.0.0.1:11434/api/tags` failed with a local connection error. Therefore no actual model inventory, active inference, streaming answer, or model-download size can be claimed.
- Current Ollama source rejects Modelfile `ADAPTER` instructions. Do not assume a PEFT adapter can be loaded directly into Ollama. Use a compatible PEFT/Transformers runtime or merge/export to a supported full-model format, and verify the resulting runtime before activation. See [Ollama create implementation](https://github.com/ollama/ollama/blob/main/cmd/cmd.go) and [PEFT checkpoint/merge guidance](https://huggingface.co/docs/peft/developer_guides/checkpoint).

## Baseline regression

The full suite was run with the project `.venv` and an isolated temporary directory because `pytest.ini` pins its default temp path to `.pytest_tmp`, which Windows denied while pytest was cleaning it. Command:

```text
.venv\\Scripts\\python.exe -m pytest tests/ -q --tb=short -p no:cacheprovider --basetemp %TEMP%\\pytest_tanya_completion_20261112_baseline
```

Baseline: **388 passed, 2 failed, 5 subtests passed**. The two failures are in `tests/test_sinta_batch_worker.py`:

1. `test_screening_cache_key_is_specific_to_each_article_doi` expects `sinta-parser:v4|`; current source returns `sinta-parser:v5|`.
2. `test_sinta_status_labels_distinguish_lookup_outcomes` expects `SINTA 2`; current source returns `Perlu Diperiksa Ulang` for the reviewed fixture.

These two failures were already recorded in the pre-completion implementation report and remain present in the unchanged SINTA files. They are the baseline, not Tanya DORA regressions; investigate the intended SINTA contract before changing them.

## Definition-of-done assessment at audit start

| Requirement | Initial status |
| --- | --- |
| Live Ollama/model inference and timed model test | **Blocked / not verified** |
| Real PDF/DOCX parsing and grounded model answer | **Blocked / not tested** in `.venv` |
| Adapter compatibility and activation in inference runtime | **Not integrated** |
| Six-function Management Studio | **Not implemented** |
| Curated verified TDID run, tokenizer, training loss, checkpoint and reload | **Blocked** by absent model, hardware validation and training dependencies |
| Five-condition TDEB comparison on the same questions | **Not run**; existing engine accepts real callbacks, but only proxy tests exist |
| Full regression | **Tested baseline**, with the two documented SINTA failures |

Do not report the Definition of Done as complete until the blocked runtime, document, training, adapter, and model-comparison checks are performed on a compatible machine.
