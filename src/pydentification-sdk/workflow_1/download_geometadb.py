import psutil
import requests
from tqdm import tqdm
import gc
import gzip
import subprocess
from pathlib import Path
from typing import Optional, List
from contextlib import contextmanager


class GeoMetaDBDownloadError(Exception):
    """Custom exception for GeoMetaDB download failures"""
    pass


class GeoMetaDBDownloader:
    """Downloads and extracts the GEOmetadb SQLite database with memory-safe streaming."""
    
    DEFAULT_URLS = [
        "https://gbnci.cancer.gov/geo/GEOmetadb.sqlite.gz",
        "https://starbuck1.s3.amazonaws.com/sradb/GEOmetadb.sqlite.gz",
    ]
    
    def __init__(self, 
        urls: Optional[List[str]] = None,
        output_dir: str = ".",
        db_filename: str = "GEOmetadb.sqlite",
        compressed_filename: str = "GEOmetadb.sqlite.gz"):
        """
        Initialize the downloader.
        
        Args:
            urls: List of URLs to try for downloading. Uses default if None.
            output_dir: Directory to save files to
            db_filename: Name for the final database file
            compressed_filename: Name for the temporary compressed file
        """
        self.urls = urls or self.DEFAULT_URLS
        self.output_dir = Path(output_dir)
        self.db_filename = db_filename
        self.compressed_filename = compressed_filename
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set chunk size based on available memory
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        self.chunk_size = 4096 if available_gb < 4 else 65536
        
        if available_gb < 4:
            print("⚠️  Warning: Low memory detected. Using conservative chunk sizes.")
    
    @property
    def db_path(self) -> Path:
        """Path to the final database file"""
        return self.output_dir / self.db_filename
    
    @property
    def compressed_path(self) -> Path:
        """Path to the temporary compressed file"""
        return self.output_dir / self.compressed_filename
    
    def download(self) -> str:
        """
        Download and extract the GEOmetadb database.
        
        Returns:
            str: Path to the downloaded database file
            
        Raises:
            GeoMetaDBDownloadError: If download fails from all sources
        """
        # Check if database already exists
        if self.db_path.exists():
            file_size = self.db_path.stat().st_size
            print(f"GEOmetadb.sqlite already exists ({file_size / (1024**3):.1f} GB)")
            return str(self.db_path)
        
        # Try each URL until one succeeds
        for url in self.urls:
            try:
                print(f"Trying: {url}")
                
                if self._download_compressed_file(url):
                    if self._decompress_file():
                        self._cleanup_compressed_file()
                        
                        final_size = self.db_path.stat().st_size
                        print(f"✅ Download complete! Final size: {final_size / (1024**3):.1f} GB")
                        return str(self.db_path)
                    else:
                        print("❌ Decompression failed")
                        
            except requests.RequestException as e:
                print(f"Network error with {url}: {e}")
                continue
            except OSError as e:
                print(f"File system error with {url}: {e}")
                continue
            except Exception as e:
                print(f"Unexpected error with {url}: {e}")
                continue
        
        raise GeoMetaDBDownloadError("Could not download GEOmetadb from any mirror")
    
    def _download_compressed_file(self, url: str) -> bool:
        """
        Download the compressed file with progress bar.
        
        Args:
            url: URL to download from
            
        Returns:
            bool: True if download successful
        """
        try:
            # Get file size for progress bar
            response = requests.head(url, timeout=10)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            # Download with streaming and progress bar
            with requests.get(url, stream=True, timeout=30) as response:
                response.raise_for_status()
                
                with open(self.compressed_path, "wb") as f:
                    with tqdm(
                        desc="Downloading",
                        total=total_size,
                        unit='B',
                        unit_scale=True,
                        unit_divisor=1024,
                    ) as pbar:
                        for chunk in response.iter_content(chunk_size=self.chunk_size):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
            
            return True
            
        except requests.RequestException as e:
            print(f"Download failed: {e}")
            return False
        except OSError as e:
            print(f"File write error: {e}")
            return False
    
    def _decompress_file(self) -> bool:
        """
        Decompress the downloaded file using memory-safe streaming.
        
        Returns:
            bool: True if decompression successful
        """
        try:
            return self._decompress_with_python()
        except MemoryError:
            print("❌ MemoryError during decompression. Trying system gzip...")
            return self._decompress_with_system_gzip()
        except Exception as e:
            print(f"Python decompression failed: {e}")
            return self._decompress_with_system_gzip()
    
    def _decompress_with_python(self) -> bool:
        """
        Decompress using Python's gzip module with streaming.
        
        Returns:
            bool: True if successful
        """
        compressed_size = self.compressed_path.stat().st_size
        print(f"Decompressing {compressed_size / (1024**2):.0f} MB file...")
        
        with gzip.open(self.compressed_path, 'rb') as f_in:
            with open(self.db_path, 'wb') as f_out:
                with tqdm(
                    desc="Extracting",
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as pbar:
                    
                    bytes_written = 0
                    
                    while True:
                        chunk = f_in.read(self.chunk_size)
                        if not chunk:
                            break
                        
                        f_out.write(chunk)
                        bytes_written += len(chunk)
                        pbar.update(len(chunk))
                        
                        # Periodic garbage collection for memory management
                        if bytes_written % (self.chunk_size * 1000) == 0:
                            gc.collect()
        
        return True
    
    def _decompress_with_system_gzip(self) -> bool:
        """
        Fallback decompression using system gzip command.
        
        Returns:
            bool: True if successful
        """
        try:
            print("Trying system gzip command...")
            
            with open(self.db_path, 'wb') as f_out:
                process = subprocess.Popen(
                    ['gzip', '-dc', str(self.compressed_path)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                # Stream the output
                while True:
                    chunk = process.stdout.read(self.chunk_size)
                    if not chunk:
                        break
                    f_out.write(chunk)
                
                process.wait()
                
                if process.returncode == 0:
                    print("✅ System gzip decompression successful")
                    return True
                else:
                    error_msg = process.stderr.read().decode()
                    print(f"❌ System gzip failed: {error_msg}")
                    return False
                    
        except FileNotFoundError:
            print("❌ System gzip command not available")
            return False
        except OSError as e:
            print(f"❌ System gzip failed: {e}")
            return False
    
    def _cleanup_compressed_file(self) -> None:
        """Remove the temporary compressed file"""
        try:
            if self.compressed_path.exists():
                self.compressed_path.unlink()
        except OSError as e:
            print(f"Warning: Could not remove compressed file: {e}")
    
    @contextmanager
    def temporary_download(self):
        """
        Context manager that ensures cleanup even if download fails.
        
        Usage:
            downloader = GeoMetaDBDownloader()
            with downloader.temporary_download():
                db_path = downloader.download()
                # Use db_path...
            # Compressed file is automatically cleaned up
        """
        try:
            yield self
        finally:
            self._cleanup_compressed_file()