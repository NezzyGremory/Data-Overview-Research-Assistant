# Real TDID and Training Validation Record

## Small reviewed dataset

`examples/tdid_trial_verified.jsonl` contains six hand-reviewed statements traceable to the local knowledge-base and fine-tuning guides. Each record is marked `verified` and names its source document and independent source group. The records cover local storage, FTS5 retrieval, scanned-PDF limits, local-only base loading, test-split isolation, and adapter merge/import. This sample checks data plumbing; it does not make a model knowledgeable or measure model quality.

## Executed validation

The fixture was parsed by `TDIDDatasetBuilder.from_jsonl` and validated with **zero issues**. Source-group splitting produced train/validation/test sizes of **4/1/1**; all six records were verified. A final snapshot was saved and reloaded in a temporary SQLite store, and `export_workspace()` produced a 3,180-byte ZIP. No runtime `data/` files were touched. The training workspace export is also covered by unit tests to include verified split JSONL, config/manifest, and readme while omitting model weights.

The `.venv` does not have PyTorch, Transformers, Datasets, PEFT, TRL, Accelerate, or bitsandbytes, and there is no compatible local base checkpoint. Therefore the following **were not tested**: tokenizer load/tokenization, a trainer optimizer step, computed loss, checkpoint creation from a real run, model reload, or evaluation inference. No training success or model capability is claimed. Run the UI preflight and repeat these checks in a separately provisioned training environment; if hardware is unsuitable, export the portable workspace.
