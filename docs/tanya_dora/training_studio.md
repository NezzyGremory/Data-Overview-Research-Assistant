# Fine-Tuning Studio

## Studio integration

**Tanya DORA Studio → Fine-Tuning** uses the existing `FineTuningEngine` and TDID version store. It provides base-model and saved-dataset selection, SFT/LoRA configuration, a QLoRA option, preflight, start/stop, actual callback progress, loss visualization, checkpoint/log locations, training history, and portable workspace export. Progress is emitted by the trainer callback; the UI does not synthesize a training run. Checkpoints, `training_manifest.json`, and a real JSONL progress log are surfaced when they exist.

Preflight checks split integrity, local model files, required dependencies, hardware, and disk estimates. It blocks before trainer start when required packages/hardware are missing and displays the dependency issue. QLoRA additionally requires compatible CUDA and bitsandbytes.

## Dataset policy and external workspace

Unverified examples are excluded from a final TDID split. A final version cannot be saved unless all included records are verified and valid; the split also requires independent groups and prevents duplicate/prompt/source leakage. `export_workspace()` packages the verified train/validation/test files, manifest, config, and instructions without bundling foundation-model weights.

The small reviewed fixture is [tdid_trial_verified.jsonl](examples/tdid_trial_verified.jsonl). It contains six short records grounded in the local Tanya DORA documentation. It is a pipeline smoke-test fixture, not sufficient training data and not evidence of expertise. It was parsed and validated with zero issues, split by source into train/validation/test (4/1/1), saved and reloaded as a final version in a temporary store, then exported as a 3,180-byte portable workspace ZIP without modifying runtime user data.

## Real training status

The installed `.venv` has none of `torch`, `transformers`, `datasets`, `peft`, `trl`, `accelerate`, or `bitsandbytes`, and no local base checkpoint is available in the project workspace. Dataset validation and workspace export are testable, but tokenizer loading, optimizer steps, loss, checkpoint save, model reload, and generated-answer evaluation were not run on this machine. No progress or model quality is claimed. Install a compatible training environment and use the preflight before starting a real run.
