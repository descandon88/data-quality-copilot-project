"""
Downloads the Olist Brazilian E-Commerce dataset (Kaggle: olistbr/brazilian-ecommerce)
straight into data/raw/, using the Kaggle API. Scripting this rather than clicking
through the website means the whole data-acquisition step is reproducible from a
clean checkout — closer to what a real ingestion pipeline's "fetch source data"
step would look like.

Requires KAGGLE_USERNAME and KAGGLE_KEY in .env (get them from
kaggle.com/settings -> API -> Create New Token). The kaggle package reads these
from the environment automatically — no kaggle.json file needed.

Run inside the app container:
    docker compose exec app python scripts/api.py
"""
import os
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
DATASET = "olistbr/brazilian-ecommerce"


def main():
    if not os.environ.get("KAGGLE_USERNAME") or not os.environ.get("KAGGLE_KEY"):
        raise SystemExit(
            "Missing KAGGLE_USERNAME / KAGGLE_KEY. Get a token at "
            "kaggle.com/settings -> API -> Create New Token, then add both "
            "values to your .env file and restart the app container."
        )

    # Imported here, not at module level: the kaggle package validates
    # credentials on import, so importing before the env-var check above
    # would produce a confusing error instead of the message above.
    from kaggle.api.kaggle_api_extended import KaggleApi

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()
    print(f"Downloading {DATASET} -> {RAW_DIR} ...")
    api.dataset_download_files(DATASET, path=str(RAW_DIR), unzip=True, quiet=False)

    csvs = sorted(RAW_DIR.glob("*.csv"))
    print(f"\nDone. {len(csvs)} CSV files in {RAW_DIR}:")
    for f in csvs:
        print(f"  {f.name}")


if __name__ == "__main__":
    main()