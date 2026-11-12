# Tanya DORA Adapter Runtime

## Runtime path

Ollama is the active chat runtime. A PEFT LoRA adapter is not treated as an Ollama model. On current Ollama source, Modelfile `ADAPTER` instructions are rejected, so the Studio offers this explicit conversion path:

1. Validate adapter hash/metadata and the structural base-model fingerprint.
2. Check PEFT type, task type, model type, target modules, local weight files, training quantization, supported Ollama import family, and required Python packages.
3. Require a local unquantized base folder and user confirmation.
4. Merge the adapter into the base using `merge_and_unload(safe_merge=True)`, save a full Safetensors model and tokenizer locally, then call `ollama create` on that folder.
5. Only after a successful import, request a second explicit model activation and record it in local activation history.

Ollama documents directory-based Safetensors import at [Importing a model](https://docs.ollama.com/import). PEFT documents merging and saving at [PEFT checkpoint guidance](https://huggingface.co/docs/peft/developer_guides/checkpoint). Supported model architectures and Ollama behavior can change; the compatibility report is intentionally restrictive.

## Management controls

The Adapter Registry supports ZIP import/export, integrity checking, compatibility inspection, conversion/import, activation, deactivation to a selected local base, model-switch history, and rollback. Conversion does not change the active model. Activation is recorded only after explicit confirmation and successful Ollama import. If merge/import fails, the current model remains selected.

The chat page receives an activation signal and selects the newly activated local Ollama tag. On page open, a persisted active model is restored only if it remains installed. A missing active tag leaves the model unselected rather than choosing an arbitrary alternative.

## Compatibility boundaries

- Quantized/4-bit base checkpoints are blocked from PEFT merging; QLoRA training produces an adapter but still needs a compatible full-precision/local base for merge.
- Only causal-LM LoRA PEFT adapters with required local files pass validation. Target module names are checked by PEFT against instantiated base layers during merge; the metadata-only report states that limitation.
- `runtime_ready` requires the local merge dependencies and Ollama CLI; successful compatibility inspection alone does not mean the runtime is ready.
- The import path is restricted to Ollama-supported Safetensors model families. A successful merge does not guarantee Ollama supports every architecture or tokenizer configuration.
- Base fingerprints are structural (config digest plus weight paths/sizes), not a cryptographic digest of every base weight.
- Merge can require substantial disk/RAM. Preflight estimates disk and reports current free space; model loading may still fail on hardware constraints.

## Current validation status

Compatibility, confirmation, registry history, rollback, and inactive-before-import behavior have automated tests. The environment has no Ollama CLI/server and lacks PyTorch, Transformers, PEFT, and Accelerate. Therefore no actual merge, Ollama import, adapter inference, or model activation has been verified here. Do not describe a registered adapter as an active runtime model until those steps succeed on a compatible machine.
