# Ollama Verification and Model Test Panel

## What the Studio now does

The Model Manager uses the existing loopback-only `OllamaLLMManager` to check the local endpoint, enumerate `/api/tags`, and query `/api/ps` for running models. The inference panel lets a user choose an installed model, stream a response, stop generation, and see elapsed time or the error. It never pulls a model automatically. The Tanya DORA chat page also refuses to silently select the first installed model; an active model must be selected explicitly.

The client has timeout, cancellation, unavailable-model, and HTTP error handling. A selected model is checked against the local inventory before a request is sent.

## Verification on this machine

**Implemented / integrated:** status, inventory, running-model query, streamed test panel, Stop, timer, and error display. The panel is wired to the same manager used by Tanya DORA.

**Tested:** automated client/UI tests use mocked HTTP responses, including stream and cancellation paths. Studio Qt construction and the dedicated page screenshot were checked.

**Not verified:** no `ollama` executable or server was available at `127.0.0.1:11434`, and the local model inventory was unavailable. No inference, token stream, model latency, or real model answer was produced. The screenshots show the unavailable state, not a successful model run.

## Install and storage guidance

Install Ollama manually, start its local server, then install an exact model tag through Ollama. Tanya DORA does not download models. Model size varies by tag and quantization. The current offline UI tells users to inspect the exact tag and free disk space in Ollama Library before installing, but it does **not** fetch and display a numeric pre-download size. Exact in-app size preflight remains unimplemented and requires a reachable registry/catalog source.

Once the server is running, open **Tanya DORA Studio → Model Manager**, refresh the connection, select the installed model in the inference panel, and run a test question. Activate it for chat only after reviewing the result and confirming the model switch.

## Ollama adapter format

Current Ollama source rejects Modelfile `ADAPTER` directives. The compatible path implemented here merges a PEFT LoRA adapter into a local base model, then imports the full Safetensors directory with a local `FROM` Modelfile. This path requires PyTorch, Transformers, PEFT, and an Ollama version that supports that Safetensors architecture. No remote model pull is performed during conversion. See [Ollama create implementation](https://github.com/ollama/ollama/blob/main/cmd/cmd.go), [Ollama Safetensors import instructions](https://docs.ollama.com/import), and [PEFT merge guidance](https://huggingface.co/docs/peft/developer_guides/checkpoint).
