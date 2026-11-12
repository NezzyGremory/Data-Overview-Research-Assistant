# AI Evaluation Studio (TDEB)

## Evaluation flow

**Tanya DORA Studio → AI Evaluation** imports a separate JSONL test benchmark, validates cases, requires a selected TDID version, and saves a snapshot only after an independent-prompt/source leakage check. The same held-out benchmark version can be run against five target conditions:

1. Base model.
2. Base model + RAG.
3. Fine-tuned Ollama model.
4. Fine-tuned model + RAG.
5. Fine-tuned model + RAG + DORA tools.

The UI persists every case answer, citation/tool output, error, latency, run status, and before/after hardware snapshot. Runs can be reviewed in the results table, exported as JSON, and sent to the human-review dialog. Review scores cover factual accuracy, instruction following, reasoning, coding, mathematics, data science, research methodology, citation accuracy, tool calling, latency quality, observed hallucination, and notes.

## Metrics and interpretation

Exact match, required-term recall, citation ID recall, and expected-tool coverage are automated lexical proxies. They do not establish factual accuracy, reasoning ability, citation faithfulness, tool correctness, or hallucination rate. Human review is required for those judgments. Never populate accuracy scores manually or infer them from latency.

## Current status

Leakage checks, same-snapshot target execution, persistence, export, and review storage are covered by service tests; the Qt page and all six Studio tabs have an offline construction test. No real five-condition comparison was possible: Ollama and installed models are absent. A base and distinct fine-tuned model must both be installed and the benchmark must be imported against a separate TDID version before live evaluation. The current environment has not produced any TDEB quality or latency result.
