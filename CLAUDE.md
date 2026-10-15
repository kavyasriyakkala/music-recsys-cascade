# Yambda Cascade: project rules

## Goal
A 3-stage recommender (retrieval -> L1 -> L2) on Yambda-50M (yandex/yambda on Hugging Face) that predicts
which tracks a user will Listen+ (played_ratio_pct > 50) during the day after a cutoff.
It studies long-history strategies and measures per-stage quality, latency, and GPU efficiency.
This is an offline-evaluation portfolio project. No online lift, no ads claims.

## Non-negotiable rules
1. NEVER fabricate results, metrics, or benchmark numbers. All numbers come from real runs written to reports/*.json.
2. NO silent fallbacks. If data, config, or an assumption is wrong, fail loudly with a clear error. Do not "gracefully" substitute defaults that change results.
3. Every assumption about the real dataset goes in DECISIONS.md, marked VERIFIED or UNVERIFIED. Synthetic data only mimics the schema; it proves code correctness, not data properties.
4. Leakage is the top risk. Every training/eval example is a snapshot: cutoff time c, inputs use only events with timestamp < c, labels come from [c, c+1 day). Enforce this in code AND tests.
5. Stage-stacked splits: retriever, L1/L2, validation, and test windows must not overlap in label periods (see src/yc/splits).
6. Every script supports --smoke (tiny synthetic data, CPU, under ~60 seconds, full pipeline end to end).
7. Code must run on CPU or CUDA via one device setting. No faiss-gpu. Exact retrieval uses torch matmul; faiss-cpu is only for the ANN comparison.
8. Python 3.11, torch, polars or pyarrow for data, pytest, ruff. Type hints on public functions. No notebooks for core logic.
9. Results JSON must include: git commit hash, config, seed, dataset scale, timestamp.
10. Keep AI_LOG.md updated: what AI wrote, how it was verified, bugs found.

## Dataset facts (from the dataset card, still verify on real data)
- Flat files sorted by (uid, timestamp). timestamp = 5-second units. is_organic: 1 organic, 0 recommendation-driven.
- multi_event columns: uid, item_id, timestamp, is_organic, event_type (listen/like/dislike/unlike/undislike), played_ratio_pct, track_length_seconds.
- embeddings.parquet: item_id, embed, normalized_embed (covers only part of the catalog).
- Whether timestamps are global or per-user deltas is UNVERIFIED. Split code must not assume it silently.

## Repo layout
configs/ src/yc/{data,splits,features,models,train,eval,serve}/ scripts/ tests/ reports/ DECISIONS.md AI_LOG.md README.md Makefile