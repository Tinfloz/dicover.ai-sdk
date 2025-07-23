import os
import requests
import GEOparse
from pathlib import Path
from typing import Optional
import time
import pandas as pd

class DifferentialExpressionDataDownloader:
    @staticmethod
    def fetch_from_geo(
        gse_id: str, 
        dest_dir: str = "./geo_data",
        timeout: int = 300,
        chunk_size: int = 8192,
        force_redownload: bool = False,
        show_progress: bool = True
    ):
        # Validate GSE ID format
        if not gse_id.startswith('GSE') or not gse_id[3:].isdigit():
            raise ValueError(f"Invalid GSE ID format: {gse_id}. Expected format: GSE followed by digits.")
        # Prepare URL components
        gse_numeric_part = gse_id[3:]
        if len(gse_numeric_part) >= 3:
            gse_prefix = "GSE" + gse_numeric_part[:-3] + "nnn"
        else:
            gse_prefix = "GSEnnn"  # For very short GSE IDs
        base_url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{gse_prefix}/{gse_id}/soft"
        file_name = f"{gse_id}_family.soft.gz"
        file_url = f"{base_url}/{file_name}"
        dest_path = Path(dest_dir)
        dest_path.mkdir(parents=True, exist_ok=True)
        local_path = dest_path / file_name
        if force_redownload or not local_path.exists():
            print(f"Downloading {gse_id} from: {file_url}")
            
            try:
                with requests.Session() as session:
                    session.headers.update({
                        'User-Agent': 'Python-GEO-Downloader/1.0'
                    })
                    response = session.get(
                        file_url, 
                        stream=True, 
                        timeout=timeout,
                        allow_redirects=True
                    )
                    response.raise_for_status()
                    
                    total_size = None
                    if 'content-length' in response.headers:
                        total_size = int(response.headers['content-length'])
                        print(f"File size: {total_size / (1024*1024):.1f} MB")
                    
                    downloaded_size = 0
                    start_time = time.time()
                    
                    with open(local_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=chunk_size):
                            if chunk:
                                f.write(chunk)
                                downloaded_size += len(chunk)
                                
                                # Show progress if requested
                                if show_progress and total_size:
                                    progress = (downloaded_size / total_size) * 100
                                    elapsed_time = time.time() - start_time
                                    speed = downloaded_size / (1024 * 1024) / elapsed_time if elapsed_time > 0 else 0
                                    print(f"\rProgress: {progress:.1f}% ({downloaded_size / (1024*1024):.1f} MB) - Speed: {speed:.1f} MB/s", end='', flush=True)
                    
                    if show_progress:
                        print()
                        
                    print(f"Successfully downloaded to: {local_path}")
                    print(f"Downloaded {downloaded_size / (1024*1024):.1f} MB in {time.time() - start_time:.1f} seconds")
                    
            except requests.exceptions.RequestException as e:
                if local_path.exists():
                    local_path.unlink() 
                raise requests.RequestException(f"Failed to download {file_url}: {str(e)}")
            
            except Exception as e:
                if local_path.exists():
                    local_path.unlink() 
                raise Exception(f"Unexpected error during download: {str(e)}")
                
        else:
            print(f"File already exists: {local_path}")
            file_size = local_path.stat().st_size / (1024 * 1024)
            print(f"File size: {file_size:.1f} MB")

        # Verify file exists and has content
        if not local_path.exists():
            raise FileNotFoundError(f"Downloaded file not found: {local_path}")
        
        if local_path.stat().st_size == 0:
            local_path.unlink()
            raise Exception(f"Downloaded file is empty: {local_path}")
        print("Parsing file with GEOparse...")
        try:
            gse = GEOparse.get_GEO(filepath=str(local_path))
            print(f"Successfully parsed: {gse.name}")
            print(f"Number of samples: {len(gse.gsms)}")
            print(f"Number of platforms: {len(gse.gpls)}")
            return gse
            
        except Exception as e:
            raise Exception(f"Failed to parse GEO file {local_path}: {str(e)}")

    @staticmethod
    def get_file_info(gse_id: str, dest_dir: str = "./geo_data") -> Optional[dict]:
        file_name = f"{gse_id}_family.soft.gz"
        local_path = Path(dest_dir) / file_name
        
        if local_path.exists():
            stat = local_path.stat()
            return {
                'path': str(local_path),
                'size_mb': stat.st_size / (1024 * 1024),
                'modified_time': time.ctime(stat.st_mtime),
                'exists': True
            }
        return None
    
    @staticmethod
    def get_expression_matrix(gse):
        """
        Constructs a gene expression matrix from GSM tables in the GSE object.
        Rows = gene identifiers, Columns = sample GSM IDs.
        """
        gene_expr = {}

        for gsm_id, gsm in gse.gsms.items():
            table = gsm.table
            if table is None or table.empty:
                continue

            try:
                genes = table.iloc[:, 0].astype(str)
                values = pd.to_numeric(table.iloc[:, 1], errors="coerce")
                gene_expr[gsm_id] = pd.Series(values.values, index=genes)
            except Exception as e:
                print(f"Skipping {gsm_id} due to parsing error: {e}")
                continue

        if not gene_expr:
            raise ValueError("No valid expression tables found in GSE object.")

        expr_df = pd.DataFrame(gene_expr)
        expr_df = expr_df.dropna()
        return expr_df