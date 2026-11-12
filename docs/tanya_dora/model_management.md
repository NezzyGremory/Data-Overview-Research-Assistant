# Model Management

**Tanya DORA Studio → Model Manager** shows the Ollama connection, installed model inventory, model family/parameter/quantization/size metadata, and running models. Its inference panel streams a selected installed model, supports Stop, reports elapsed time/errors, and does not auto-pull. When Ollama is absent it displays setup guidance and tells the user to verify the chosen tag's disk size before installation. Exact numeric pre-download size lookup is not yet built (see [Ollama verification](ollama_verification.md)).

The local active chat model is stored separately from the adapter registry. Selecting a base model, activating a merged adapter, deactivating to a base, or rolling back each requires explicit user confirmation. Activation history keeps the prior model metadata for rollback. Rollback is blocked if the previous model tag is no longer installed. The Tanya DORA chat selector restores a persisted choice only when it is in the current local inventory.

The adapter table includes registered versions and provenance. Users can import/export adapter ZIPs, inspect compatibility, and request conversion to a full local Safetensors model for Ollama. The registered adapter alone is not loadable by the Ollama chat path.

Automated tests cover model selection/history/rollback, adapter archive integrity and compatibility, and Qt page construction. No live model list or inference was available in this environment, so no model benchmark, model-switch inference, or rollback against a real Ollama server is verified.
