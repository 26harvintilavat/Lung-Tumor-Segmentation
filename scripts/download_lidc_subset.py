"""
Download a small subset of the LIDC-IDRI dataset from TCIA.

This scripts:
1. Quiries all CT series in LIDC-IDRI
2. Selects the first N series (configurable)
3. Downloads only those series
"""

import tcia_utils
from tcia_utils import nbia
import os

collection = "LIDC-IDRI"
num_series = 20
download_dir = "data/raw/LIDC_subset"

def main():
    os.makedirs(download_dir, exist_ok=True)

    print("Fetching series metadata...")
    series_list = nbia.getSeries(collection=collection)

    print(f"Total series found: {len(series_list)}")

    selected_series = series_list[:num_series]

    print(f"Downloading {len(selected_series)} series...")
    nbia.downloadSeries(selected_series, download_dir)

    print("Download completed.")

if __name__=="__main__":
    main()