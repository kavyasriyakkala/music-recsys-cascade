from huggingface_hub import hf_hub_download
import os

LOCAL_DIR = os.environ.get("YAMBDA_DIR", "/scratch/ky2751/yambda/50m")

files = [
    "flat/50m/multi_event.parquet",
    "flat/50m/listens.parquet",
    "flat/50m/likes.parquet",
    "sequential/50m/multi_event.parquet",
    "embeddings.parquet",
    "artist_item_mapping.parquet",
]

for f in files:
    print(f"Downloading {f}...", flush=True)
    hf_hub_download(
        repo_id="yandex/yambda",
        repo_type="dataset",
        filename=f,
        local_dir=LOCAL_DIR,
    )
    print(f"  done.", flush=True)

print("All files downloaded.")
