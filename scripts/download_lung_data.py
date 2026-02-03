"""
TCIA Lung Tumor Dataset Downloader

This script downloads a specified number of CT series from The Cancer Imaging Archive (TCIA) for lung tumor segmentation projects.

Author: Harvin Tilavat
Date: 02/02/2026

"""

from tcia_utils import nbia
import os
import pandas as pd
from typing import Optional
from configs import config


def fetch_series_info(collection: str = config.COLLECTION_NAME) -> pd.DataFrame:
    """
    Fetch series information from TCIA collection.
    
    Args:
        collection (str): Name of the TCIA collection
        
    Returns:
        pd.DataFrame: DataFrame containing series information
    """
    print(f"Fetching series information from {collection}...")
    series_df = nbia.getSeries(collection=collection, format="df")
    print(f"Total series available: {len(series_df)}")
    print(f"Available modalities: {series_df['Modality'].unique()}")
    return series_df


def filter_ct_series(series_df: pd.DataFrame, 
                     num_series: int = config.NUM_SERIES_TO_DOWNLOAD,
                     modality: str = config.MODALITY_FILTER) -> pd.DataFrame:
    """
    Filter series by modality from the dataset.
    
    Args:
        series_df (pd.DataFrame): DataFrame containing all series
        num_series (int): Number of series to select
        modality (str): Modality to filter (default: CT)
        
    Returns:
        pd.DataFrame: Filtered DataFrame with specified modality
    """
    filtered_series = series_df[series_df['Modality'] == modality].head(num_series)
    print(f"\nSelected {len(filtered_series)} {modality} series")
    print(f"Unique patients: {filtered_series['PatientID'].nunique()}")
    
    if len(filtered_series) < num_series:
        print(f"Warning: Only {len(filtered_series)} {modality} series available (requested {num_series})")
    
    return filtered_series


def download_series(ct_series: pd.DataFrame, 
                    download_path: str = config.DOWNLOAD_DIRECTORY,
                    manifest_filename: str = config.MANIFEST_FILENAME) -> bool:
    """
    Download selected series from TCIA.
    
    Args:
        ct_series (pd.DataFrame): DataFrame with selected series
        download_path (str): Directory to save downloaded data
        manifest_filename (str): Name of the manifest CSV file
        
    Returns:
        bool: True if download successful, False otherwise
    """
    # Create download directory
    os.makedirs(download_path, exist_ok=True)
    print(f"\nDownload path: {download_path}")
    
    try:
        # Try downloading with dataframe directly
        print("Attempting download with dataframe input...")
        nbia.downloadSeries(
            series_data=ct_series,
            path=download_path,
            input_type="df"
        )
        print("Download complete!")
        return True
        
    except Exception as e:
        print(f"Error with dataframe method: {e}")
        print("\nTrying alternative manifest method...")
        
        try:
            # Alternative: Save to CSV and use manifest
            ct_series.to_csv(manifest_filename, index=False)
            print(f"Manifest saved: {manifest_filename}")
            
            nbia.downloadSeries(
                series_data=manifest_filename,
                path=download_path,
                input_type="manifest"
            )
            print("Download complete using manifest!")
            return True
            
        except Exception as e2:
            print(f"Error with manifest method: {e2}")
            return False


def save_metadata(ct_series: pd.DataFrame, 
                  output_path: str = config.DOWNLOAD_DIRECTORY,
                  filename: str = config.METADATA_FILENAME):
    """
    Save metadata of downloaded series.
    
    Args:
        ct_series (pd.DataFrame): DataFrame with selected series
        output_path (str): Directory to save metadata
        filename (str): Name of the metadata file
    """
    metadata_file = os.path.join(output_path, filename)
    ct_series.to_csv(metadata_file, index=False)
    print(f"Metadata saved: {metadata_file}")


def print_summary(ct_series: pd.DataFrame):
    """
    Print summary statistics of the downloaded dataset.
    
    Args:
        ct_series (pd.DataFrame): DataFrame with selected series
    """
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"Total series downloaded: {len(ct_series)}")
    print(f"Unique patients: {ct_series['PatientID'].nunique()}")
    print(f"Total images: {ct_series['ImageCount'].sum()}")
    print(f"Modality: {ct_series['Modality'].unique()}")
    if 'BodyPartExamined' in ct_series.columns:
        print(f"Body parts: {ct_series['BodyPartExamined'].unique()}")
    print("=" * 60)


def main(collection: str = config.COLLECTION_NAME,
         num_series: int = config.NUM_SERIES_TO_DOWNLOAD,
         download_path: str = config.DOWNLOAD_DIRECTORY):
    """
    Main function to download lung tumor dataset.
    
    Args:
        collection (str): TCIA collection name
        num_series (int): Number of series to download
        download_path (str): Path to save downloaded data
    """
    print("=" * 60)
    print("TCIA Lung Tumor Dataset Downloader")
    print("=" * 60)
    
    # Step 1: Fetch series information
    series_df = fetch_series_info(collection)
    
    # Step 2: Filter CT series
    ct_series = filter_ct_series(series_df, num_series)
    
    # Step 3: Download series
    success = download_series(ct_series, download_path)
    
    # Step 4: Save metadata and print summary
    if success:
        save_metadata(ct_series, download_path)
        print_summary(ct_series)
        print(f"\n All operations completed successfully!")
        print(f"Data location: {os.path.abspath(download_path)}\n")
    else:
        print("\n Download failed. Please check the errors above.\n")


if __name__ == "__main__":
    # Run download with settings from config.py
    main()


    
