"""
Download a small subset of the LIDC-IDRI dataset from TCIA.

This scripts:
1. Quiries all CT series in LIDC-IDRI
2. Selects the first N series (configurable)
3. Downloads only those series
"""

import os
from tcia_utils import nbia

# ================= CONFIG =================
COLLECTION = "LIDC-IDRI"
NUM_SERIES = 20
DOWNLOAD_DIR = "data/raw/LIDC_subset"
# =========================================


def main():
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    print("Fetching series metadata...")
    series_list = nbia.getSeries(collection=COLLECTION)

    print(f"Total series found: {len(series_list)}")

    # Keep full series dictionaries
    selected_series = series_list[:NUM_SERIES]

    print(f"Downloading {len(selected_series)} series...")

    nbia.downloadSeries(
        selected_series,
        number=0,
        downloadDir=DOWNLOAD_DIR
    )

    print("Download completed successfully.")


if __name__ == "__main__":
    main()
