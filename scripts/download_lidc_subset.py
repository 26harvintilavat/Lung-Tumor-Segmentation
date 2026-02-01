"""
Download a small subset of the LIDC-IDRI dataset from TCIA.

This scripts:
1. Quiries all CT series in LIDC-IDRI
2. Selects the first N series (configurable)
3. Downloads only those series
"""

import os
from tcia_utils import nbia

COLLECTION = "LIDC-IDRI"
NUM_SERIES = 20
DOWNLOAD_DIR = "data/raw/LIDC_subset"


def main():
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    print("Fetching series metadata...")
    series_list = nbia.getSeries(collection=COLLECTION)

    print(f"Total series found: {len(series_list)}")

    selected_series = series_list[:NUM_SERIES]
    series_uids = [s["SeriesInstanceUID"] for s in selected_series]

    print(f"Downloading {len(series_uids)} series...")

    # STABLE API (tcia-utils 1.2.0)
    nbia.downloadSeries(
        seriesInstanceUid=series_uids,
        downloadDir=DOWNLOAD_DIR
    )

    print("Download completed successfully.")


if __name__ == "__main__":
    main()