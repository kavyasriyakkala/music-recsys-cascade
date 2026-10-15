"""Generate a tiny Yambda-shaped synthetic dataset for smoke tests and dev.

Timestamps are global integers in 5-second units (UNVERIFIED for real data —
see DECISIONS.md). Users have latent tastes; items belong to clusters, making
the data learnable: a model can beat pure popularity.
"""

from __future__ import annotations

import numpy as np
import polars as pl


# 1 day in 5-second units
DAY_UNITS: int = 86400 // 5  # 17280


def generate_dataset(
    n_users: int = 200,
    n_items: int = 2000,
    n_days: int = 30,
    n_latent: int = 5,
    embedding_dim: int = 64,
    embedding_coverage: float = 0.8,
    listen_plus_threshold: float = 0.5,  # Yambda dataset card: played_ratio_pct > 50
    seed: int = 42,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Return (events, embeddings, item_artist).

    events schema: uid, item_id, timestamp, is_organic, event_type,
                   played_ratio_pct, track_length_seconds
    embeddings schema: item_id, embed (list[float]), normalized_embed (list[float])
    item_artist schema: item_id, artist_id
    """
    rng = np.random.default_rng(seed)

    # --- latent structure ---
    # Each user has a taste weight vector over n_latent clusters.
    user_tastes = rng.dirichlet(np.ones(n_latent), size=n_users)  # (U, L)
    # Each item belongs to one cluster.
    item_cluster = rng.integers(0, n_latent, size=n_items)  # (I,)

    # --- item catalog ---
    item_ids = np.arange(n_items)
    n_artists = max(1, n_items // 5)
    item_artist_arr = rng.integers(0, n_artists, size=n_items)

    # --- embeddings (covering embedding_coverage fraction of items) ---
    n_embedded = int(n_items * embedding_coverage)
    embedded_items = rng.choice(item_ids, size=n_embedded, replace=False)
    raw_embeds = rng.standard_normal((n_embedded, embedding_dim)).astype(np.float32)
    norms = np.linalg.norm(raw_embeds, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normalized_embeds = raw_embeds / norms

    embeddings_df = pl.DataFrame(
        {
            "item_id": embedded_items.tolist(),
            "embed": raw_embeds.tolist(),
            "normalized_embed": normalized_embeds.tolist(),
        }
    )

    item_artist_df = pl.DataFrame(
        {
            "item_id": item_ids.tolist(),
            "artist_id": item_artist_arr.tolist(),
        }
    )

    # --- events ---
    uid_list: list[int] = []
    item_id_list: list[int] = []
    timestamp_list: list[int] = []
    is_organic_list: list[int] = []
    event_type_list: list[str] = []
    played_ratio_pct_list: list[float] = []
    track_length_list: list[float] = []

    event_types = ["listen", "like", "dislike", "unlike", "undislike"]
    # listen events are most common; others are rare reactions
    event_type_weights = np.array([0.70, 0.12, 0.10, 0.04, 0.04])
    event_type_weights /= event_type_weights.sum()

    # Each user generates events across n_days.
    # Events per user per day drawn from a Poisson with mean scaled by activity.
    for uid in range(n_users):
        taste = user_tastes[uid]  # (L,)
        # activity: 2-12 events per day on average
        daily_rate = rng.integers(2, 13)

        for day in range(n_days):
            n_events = rng.poisson(daily_rate)
            if n_events == 0:
                continue

            # Sample items with probability proportional to taste-cluster affinity.
            affinity = taste[item_cluster]  # (I,)
            affinity = affinity / affinity.sum()
            sampled_items = rng.choice(item_ids, size=n_events, p=affinity)

            # Spread events uniformly within the day.
            day_start = day * DAY_UNITS
            offsets = rng.integers(0, DAY_UNITS, size=n_events)
            timestamps = np.sort(day_start + offsets)

            for idx in range(n_events):
                iid = int(sampled_items[idx])
                ts = int(timestamps[idx])
                etype = rng.choice(event_types, p=event_type_weights)
                organic = int(rng.random() > 0.3)
                # played_ratio_pct: listen+ more likely when taste matches cluster
                cluster_match = taste[item_cluster[iid]]
                # Base probability of listen+ scales with affinity
                p_listen_plus = float(np.clip(cluster_match * n_latent, 0.1, 0.95))
                if rng.random() < p_listen_plus:
                    ratio = rng.uniform(listen_plus_threshold * 100 + 1, 100)
                else:
                    ratio = rng.uniform(0, listen_plus_threshold * 100)
                track_len = float(rng.integers(60, 300))

                uid_list.append(uid)
                item_id_list.append(iid)
                timestamp_list.append(ts)
                is_organic_list.append(organic)
                event_type_list.append(etype)
                played_ratio_pct_list.append(round(ratio, 1))
                track_length_list.append(track_len)

    events_df = pl.DataFrame(
        {
            "uid": uid_list,
            "item_id": item_id_list,
            "timestamp": timestamp_list,
            "is_organic": is_organic_list,
            "event_type": event_type_list,
            "played_ratio_pct": played_ratio_pct_list,
            "track_length_seconds": track_length_list,
        }
    ).sort(["uid", "timestamp"])

    return events_df, embeddings_df, item_artist_df
