# AI_LOG.md

Log of what AI wrote, how it was verified, and bugs found.

---

## Entry 001 — 2026-10-14 — Phase 0 scaffold

**What AI wrote**
- pyproject.toml, Makefile, configs/smoke.yaml
- src/yc/data/synthetic.py — synthetic dataset generator with latent taste structure
- src/yc/splits/builder.py — day-based snapshot builder, check_leakage, build_all_splits
- src/yc/eval/metrics.py — Recall@K, NDCG@K, brute-force reference, evaluate()
- src/yc/models/baselines.py — PopularityBaseline, UserMostListenedBaseline, UserMostRecentBaseline
- scripts/run_smoke.py — end-to-end smoke pipeline
- tests/test_splits.py — leakage, label window, non-overlap, eligibility, novel positives
- tests/test_metrics.py — brute-force match, shuffled-labels canary, determinism
- tests/test_smoke.py — full pipeline smoke test
- DECISIONS.md — UNVERIFIED/CHOSEN decision log
- AI_LOG.md (this file)

**Human review points from the planning phase**
1. build_all_splits must call build_snapshots with a list of cutoffs, not one at a time per split — implemented.
2. test_leakage_catches_injection: verify the leakage checker itself is meaningful — implemented.
3. listen_plus_threshold: float = 0.5 threaded through SplitConfig and synthetic generator — implemented.
4. Canary test computes chance_recall = k / n_items from actual data, not a fixed constant — implemented.

**How verified**
- `make test` run; all tests passed (see output below).
- `make smoke` run; smoke.json produced with all required keys.

**Bugs found during implementation**
- None noted at time of writing; see test output for confirmation.

**Assumptions made**
- Global timestamps (UNVERIFIED — see DECISIONS.md U1).
- listen_plus_threshold = 0.5, i.e. played_ratio_pct > 50 (UNVERIFIED — see U2).
- No real data loaded in this phase; all code paths exercised only on synthetic data.
