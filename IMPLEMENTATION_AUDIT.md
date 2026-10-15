# Implementation audit

The plugin implements the single-tool serial association protocol, prepared/out failure handling, event-backed exact/censored labels, frozen structured features, XGBoost normal AFT, validation-selected sigma, conditional remaining distributions, quality-gated atomic promotion, periodic batch retraining, optional joint-semantic ablation, and a scheduler-neutral leased client API.

Local tests cover the complete reducer/training/package/gate/window path, HTTP tenant isolation, streaming and nonstreaming send boundaries, late returns, cancellation, event consistency, Redis transaction wiring and training leases with fakeredis. Acceptance objects in unit tests are explicitly fabricated fixtures, not target-server evidence.

Real Redis, pinned GPU SGLang, multi-host chrony, real semantic encoder quality, online calibration and target-server P99/throughput still require runtime evidence. No proxy or synthetic experiment authorizes production publication. C2 density, KV mapping/loading and speculative benefit experiments are owned by the external integrating modules.

Run `uv sync --extra test` followed by `uv run pytest -q`. The default baseline requires no semantic model download. See INTEGRATION.md for the external contract and README.md for deployment/training/promotion commands.
