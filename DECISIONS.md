# DECISIONS.md

Design decisions and dataset assumptions for Yambda-cascade.
Every assumption about the real dataset is marked VERIFIED or UNVERIFIED.
Verified means confirmed by direct inspection of real data; Unverified means
assumed from the dataset card or not yet checked.

---

## VERIFIED

| # | Assumption | Evidence |
|---|---|---|
| V1 | Timestamps are global 5-second units (not per-user deltas) | Gate 1 on real data: only 5.11% of users have min_ts near zero; per-user deltas would show ~100%. Dataset spans 1504 days (~4.1 years) — full platform history. Split windows reference the last N days and are unchanged. |

---

## UNVERIFIED

| # | Assumption | Where Used | How to Verify |
|---|---|---|---|
| ~~U1~~ | ~~Timestamps are global integers in 5-second units~~ | Moved to VERIFIED as V1 |
| U2 | A "Listen+" is defined as played_ratio_pct > 50 (threshold = 0.5) | SplitConfig.listen_plus_threshold, synthetic label generation | Confirm against official Yambda dataset card or benchmark protocol |
| U3 | embeddings.parquet covers ~80% of the catalog | synthetic embedding_coverage = 0.8 | Count distinct item_ids in embeddings.parquet vs. multi_event |
| U4 | Daily volume: average events per user per day is in low single digits to low tens | synthetic daily_rate in [2, 12] | Compute mean events/user/day on real data |
| U5 | No official benchmark protocol (split dates, evaluation K, held-out users) exists | all eval code | Check dataset card, HuggingFace discussions, any accompanying paper |
| U6 | event_type values are exactly: listen, like, dislike, unlike, undislike | synthetic event_types list | Count distinct event_type in real data |

---

## CHOSEN

| # | Decision | Rationale |
|---|---|---|
| C1 | Day boundary = multiples of DAY_UNITS = 17280 (5-sec units) | Consistent with U1; day 0 starts at timestamp 0 |
| C2 | Snapshot cutoff = start of label day (d * DAY_UNITS) | Clean, reproducible; no time-of-day leakage |
| C3 | Novel positive = item never listened (any played_ratio_pct) before cutoff, not just never Listen+ | Stricter definition; avoids crediting re-discovery of items already heard quietly |
| C4 | min_prior_events = 50 for full runs; lowered to 5 in smoke config | 50 is enough history for a meaningful user embedding; 5 lets smoke tests run on tiny data |
| C5 | Window boundaries measured back from day N: retriever_train ≤ N-16, retriever_val = N-15, rank_train = N-14..N-3, validation = N-2, test = N | Leaves 2-day buffer between rank_train and test; retriever and ranker are trained on different label periods |
| C6 | build_snapshots uses timestamp < cutoff (strict), not <=, to avoid leakage | A timestamp exactly at cutoff would be in the label window; strict < is safe |
| C7 | Synthetic latent structure: users draw taste weights from Dirichlet(1); items assigned to one of n_latent clusters; listen+ probability scales with taste-cluster affinity | Makes data learnable — a model can beat pure popularity — while keeping generation simple |
| C8 | Synthetic timestamps are global and sorted by (uid, timestamp) before export | Matches the dataset card description of flat files sorted by (uid, timestamp) |
| C9 | No faiss-gpu; exact retrieval uses torch matmul; faiss-cpu only for ANN comparison | Reproducibility across CPU-only machines; faiss-gpu has install complexity |
