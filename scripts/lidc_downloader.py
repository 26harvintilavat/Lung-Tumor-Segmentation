"""
LIDC-IDRI Dataset Downloader - Simplified & Robust Version
Downloads CT scans from The Cancer Imaging Archive (TCIA)
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
PROJECT_ROOT = Path(__file__).resolve().parents[1]

print("Checking dependencies...")
try:
    from tcia_utils import nbia
    print("✓ tcia_utils found")
except ImportError:
    print("\n✗ ERROR: tcia_utils not installed")
    print("Please run: pip install tcia-utils")
    sys.exit(1)


class LIDCIDRIDownloader:
    """
    Simple, robust downloader for LIDC-IDRI dataset.
    
    Usage:
        downloader = LIDCIDRIDownloader(num_series=10)
        downloader.download()
    """
    
    def __init__(self, output_dir="data/raw/lung_data", num_series=None):
        """
        Args:
            output_dir: Where to save data
            num_series: How many series to download (None = all)
        """
        self.output_dir = (PROJECT_ROOT/output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_series = num_series
        
        # Metadata file
        self.metadata_file = self.output_dir / "download_log.json"
        self.downloaded = self._load_downloaded()
        
        print(f"\n{'='*60}")
        print(f"LIDC-IDRI Downloader Initialized")
        print(f"{'='*60}")
        print(f"Output: {self.output_dir.absolute()}")
        print(f"Series to download: {num_series if num_series else 'ALL'}")
        print(f"Already downloaded: {len(self.downloaded)}")
        print(f"{'='*60}\n")
    
    def _load_downloaded(self):
        """Load list of already downloaded series"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                    return set(data.get('downloaded', []))
            except:
                pass
        return set()
    
    def _save_downloaded(self):
        """Save list of downloaded series"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump({
                    'downloaded': list(self.downloaded),
                    'last_update': datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save metadata: {e}")
    
    def get_all_series(self):
        """Get all available series from LIDC-IDRI"""
        print("Fetching series from TCIA...")
        
        try:
            # Get all series directly from collection
            series_data = nbia.getSeries(
                collection="LIDC-IDRI",
                modality="CT"
            )
            
            if series_data is None:
                raise ValueError("No data returned from TCIA")
            
            print(f"✓ Received data from TCIA (type: {type(series_data).__name__})")
            
            # Convert to list of dicts
            series_list = []
            
            # Handle pandas DataFrame
            if hasattr(series_data, 'iterrows'):
                print(f"✓ Processing DataFrame with {len(series_data)} rows")
                for idx, row in series_data.iterrows():
                    try:
                        # Try different possible column names
                        series_uid = (row.get('SeriesInstanceUID') or 
                                    row.get('Series UID') or 
                                    row.get('Series Instance UID'))
                        
                        patient_id = (row.get('PatientID') or 
                                    row.get('Patient ID'))
                        
                        if series_uid and patient_id:
                            series_list.append({
                                'series_uid': str(series_uid),
                                'patient_id': str(patient_id),
                                'image_count': int(row.get('ImageCount', row.get('Number of Images', 0)))
                            })
                    except Exception as e:
                        continue
            
            # Handle list
            elif isinstance(series_data, list):
                print(f"✓ Processing list with {len(series_data)} items")
                for item in series_data:
                    try:
                        if isinstance(item, dict):
                            series_uid = (item.get('SeriesInstanceUID') or 
                                        item.get('Series UID'))
                            patient_id = (item.get('PatientID') or 
                                        item.get('Patient ID'))
                            
                            if series_uid and patient_id:
                                series_list.append({
                                    'series_uid': str(series_uid),
                                    'patient_id': str(patient_id),
                                    'image_count': int(item.get('ImageCount', 0))
                                })
                    except:
                        continue
            
            if not series_list:
                print("\n✗ ERROR: Could not extract series information")
                print(f"Data type received: {type(series_data)}")
                if hasattr(series_data, 'columns'):
                    print(f"Columns: {list(series_data.columns)}")
                if hasattr(series_data, 'head'):
                    print(f"First row:\n{series_data.head(1)}")
                raise ValueError("No series could be extracted")
            
            print(f"✓ Successfully extracted {len(series_list)} series")
            
            # Limit if requested
            if self.num_series and self.num_series < len(series_list):
                series_list = series_list[:self.num_series]
                print(f"✓ Limited to first {self.num_series} series")
            
            return series_list
            
        except Exception as e:
            print(f"\n✗ ERROR getting series: {e}")
            print(f"Error type: {type(e).__name__}")
            raise
    
    def download_series(self, series_uid, patient_id, index, total):
        """Download a single series"""
        
        # Skip if already downloaded
        if series_uid in self.downloaded:
            print(f"[{index}/{total}] ⊙ Skipping {patient_id} (already downloaded)")
            return True
        
        print(f"\n[{index}/{total}] Downloading {patient_id}")
        print(f"  Series: {series_uid[:50]}...")
        
        try:
            # Create patient directory
            patient_dir = self.output_dir / patient_id
            patient_dir.mkdir(exist_ok=True)
            
            # Download
            nbia.downloadSeries(
                series_data=[series_uid],
                input_type="list",
                path=str(patient_dir)
            )
            
            # Verify something was downloaded
            if not any(patient_dir.rglob("*.dcm")):
                print(f"  ⚠ Warning: No DICOM files found after download")
                return False
            
            # Mark as downloaded
            self.downloaded.add(series_uid)
            self._save_downloaded()
            
            print(f"  ✓ Success")
            return True
            
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            return False
    
    def download(self):
        """Main download function"""
        start_time = time.time()
        
        try:
            # Get series list
            print("\n" + "="*60)
            series_list = self.get_all_series()
            
            # Download each series
            print(f"\n{'='*60}")
            print(f"STARTING DOWNLOADS")
            print(f"{'='*60}\n")
            
            success = 0
            failed = 0
            skipped = 0
            
            for idx, series_info in enumerate(series_list, 1):
                if series_info['series_uid'] in self.downloaded:
                    skipped += 1
                    if idx % 10 == 0:  # Only print every 10th skip
                        print(f"[{idx}/{len(series_list)}] ⊙ Skipping (already have {skipped} series)...")
                    continue
                
                result = self.download_series(
                    series_info['series_uid'],
                    series_info['patient_id'],
                    idx,
                    len(series_list)
                )
                
                if result:
                    success += 1
                else:
                    failed += 1
                
                # Progress update every 5 downloads
                if idx % 5 == 0:
                    elapsed = time.time() - start_time
                    percent = (idx / len(series_list)) * 100
                    print(f"\n--- Progress: {percent:.1f}% | Success: {success} | Failed: {failed} | Skipped: {skipped} | Time: {elapsed/60:.1f}m ---\n")
            
            # Final summary
            elapsed = time.time() - start_time
            
            print(f"\n{'='*60}")
            print(f"DOWNLOAD COMPLETE")
            print(f"{'='*60}")
            print(f"Total series: {len(series_list)}")
            print(f"New downloads: {success}")
            print(f"Already had: {skipped}")
            print(f"Failed: {failed}")
            print(f"Time: {elapsed/60:.1f} minutes")
            print(f"Location: {self.output_dir.absolute()}")
            print(f"{'='*60}\n")
            
            return {
                'total': len(series_list),
                'success': success,
                'failed': failed,
                'skipped': skipped,
                'time_minutes': elapsed/60
            }
            
        except KeyboardInterrupt:
            print("\n\n⚠ Download interrupted by user")
            print("Progress has been saved. Run again to resume.")
            sys.exit(0)
        except Exception as e:
            print(f"\n✗ Download failed: {e}")
            raise


def main():
    """Command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download LIDC-IDRI dataset from TCIA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Download 10 series:
    python %(prog)s --num-series 10
  
  Download 50 series to custom directory:
    python %(prog)s --num-series 50 --output-dir ./my_data
  
  Download all available series:
    python %(prog)s
        """
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/raw/lung_data',
        help='Output directory (default: data/raw/lung_data)'
    )
    
    parser.add_argument(
        '--num-series',
        type=int,
        default=None,
        help='Number of series to download (default: all)'
    )
    
    args = parser.parse_args()
    
    # Create downloader
    downloader = LIDCIDRIDownloader(
        output_dir=args.output_dir,
        num_series=args.num_series
    )
    
    # Download
    try:
        stats = downloader.download()
        sys.exit(0)
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        print(f"\nFatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()