"""
Configuration file for TCIA dataset download.

Modify these settings according to your needs.
"""

# TCIA collection configuration
# available options: "NSCLC-Radiomics", "LUNG1", "LIDC-IDRI"
COLLECTION_NAME = "NSCLC-Radiomics"

# Number of series to download
# Set to a lower number for testing, higher for full dataset
NUM_SERIES_TO_DOWNLOAD = 1

# Download directory
# Will be created if it doesn't exist
DOWNLOAD_DIRECTORY = "data/raw/lung_data"

# Metadata filename
METADATA_FILENAME = "dataset_metadata.csv"

# Manifest filename (used in fallback method)
MANIFEST_FILENAME = "series_manifest.csv"

# filter settings
MODALITY_FILTER = "CT"

# Advanced settings
VERBOSE = True