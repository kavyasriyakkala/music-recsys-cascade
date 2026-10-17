from huggingface_hub import hf_hub_download
import os
import shutil
from pathlib import Path

LOCAL_DIR = os.environ.get("YAMBDA_DIR", "/scratch/ky2751/yambda/50m")

# Map HuggingFace path -> flat filename in LOCAL_DIR.
# multi_event from flat/50m lands as multi_event.parquet for easy discovery.
files = {
    "flat/50m/multi_event.parquet": "multi_event.parquet",
    "flat/50m/listens.parquet": "listens.parquet",
    "flat/50m/likes.parquet": "likes.parquet",
    "sequential/50m/multi_event.parquet": "sequential_multi_event.parquet",
    "embeddings.parquet": "embeddings.parquet",
    "artist_item_mapping.parquet": "artist_item_mapping.parquet",
}

Path(LOCAL_DIR).mkdir(parents=True, exist_ok=True)

for hf_path, local_name in files.items():
    dest = Path(LOCAL_DIR) / local_name
    print(f"Downloading {hf_path} -> {dest} ...", flush=True)
    tmp = hf_hub_download(
        repo_id="yandex/yambda",
        repo_type="dataset",
        filename=hf_path,
        local_dir_use_symlinks=False,
    )
    shutil.copy2(tmp, dest)
    print(f"  done.", flush=True)

print("All files downloaded.")
