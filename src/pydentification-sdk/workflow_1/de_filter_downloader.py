import os
import requests
import GEOparse
from pathlib import Path
from typing import Optional, Union
import time
import pandas as pd
import numpy as np
import gzip
import re

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
    def download_supplementary_files(
        gse_id: str,
        dest_dir: str = "./geo_data",
        file_patterns: list = None,
        force_redownload: bool = False,
        timeout: int = 300
    ):
        """
        Download supplementary files from GEO (like RNA-seq count matrices).
        
        Args:
            gse_id: GSE identifier
            dest_dir: Destination directory
            file_patterns: List of filename patterns to download (e.g., ['*counts*', '*matrix*'])
                          If None, downloads all supplementary files
            force_redownload: Whether to redownload existing files
            timeout: Request timeout
        
        Returns:
            List of downloaded file paths
        """
        gse_numeric_part = gse_id[3:]
        if len(gse_numeric_part) >= 3:
            gse_prefix = "GSE" + gse_numeric_part[:-3] + "nnn"
        else:
            gse_prefix = "GSEnnn"
        
        # Try to get supplementary file URLs from the FTP directory
        ftp_base = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{gse_prefix}/{gse_id}/suppl/"
        
        dest_path = Path(dest_dir)
        dest_path.mkdir(parents=True, exist_ok=True)
        
        downloaded_files = []
        
        # Common RNA-seq file patterns if none specified
        if file_patterns is None:
            file_patterns = ['*counts*', '*matrix*', '*fpkm*', '*tpm*', '*expression*']
        
        # Try some common supplementary file names for RNA-seq
        common_rnaseq_files = [
            f"{gse_id}_counts.csv.gz",
            f"{gse_id}_totalRNA_counts.csv.gz",
            f"{gse_id}_count_matrix.csv.gz",
            f"{gse_id}_expression_matrix.csv.gz",
            f"{gse_id}_gene_counts.csv.gz",
            f"{gse_id}_raw_counts.csv.gz",
            f"{gse_id}_counts.txt.gz",
            f"{gse_id}_fpkm.csv.gz",
            f"{gse_id}_tpm.csv.gz"
        ]
        
        print(f"Attempting to download supplementary files for {gse_id}...")
        
        for filename in common_rnaseq_files:
            # Check if filename matches any of the patterns
            if file_patterns:
                matches_pattern = any(
                    re.search(pattern.replace('*', '.*'), filename.lower()) 
                    for pattern in file_patterns
                )
                if not matches_pattern:
                    continue
            
            file_url = ftp_base + filename
            local_path = dest_path / filename
            
            if not force_redownload and local_path.exists():
                print(f"File already exists: {local_path}")
                downloaded_files.append(local_path)
                continue
            
            try:
                print(f"Trying to download: {file_url}")
                response = requests.get(file_url, timeout=timeout, stream=True)
                response.raise_for_status()
                
                # Download the file
                with open(local_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                file_size = local_path.stat().st_size / (1024 * 1024)
                print(f"✓ Downloaded: {filename} ({file_size:.1f} MB)")
                downloaded_files.append(local_path)
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 404:
                    continue  # File doesn't exist, try next one
                else:
                    print(f"HTTP error downloading {filename}: {e}")
            except Exception as e:
                print(f"Error downloading {filename}: {e}")
        
        if not downloaded_files:
            print("No supplementary files found with the specified patterns.")
            print("You may need to check the GEO page manually for available files.")
        
        return downloaded_files

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
    def _detect_table_structure(table):
        """
        Detect the structure of a GSM table to determine column meanings.
        Returns a tuple of (gene_col_idx, value_col_idx, table_type)
        """
        if table is None or table.empty:
            return None, None, "empty"
        
        # Get column names
        columns = table.columns.tolist()
        n_cols = len(columns)
        
        # Check for common RNA-seq column patterns
        gene_col_idx = None
        value_col_idx = None
        
        # Look for gene identifier columns
        for i, col in enumerate(columns):
            col_lower = str(col).lower()
            if any(term in col_lower for term in ['gene', 'id', 'symbol', 'probe', 'transcript']):
                gene_col_idx = i
                break
        
        # Look for expression value columns
        for i, col in enumerate(columns):
            col_lower = str(col).lower()
            if any(term in col_lower for term in ['count', 'fpkm', 'tpm', 'rpkm', 'value', 'signal', 'expression']):
                value_col_idx = i
                break
        
        # If we haven't found specific columns, try to infer from data types
        if gene_col_idx is None or value_col_idx is None:
            # Look for the first string-like column (likely genes)
            # and the first numeric column (likely values)
            for i in range(n_cols):
                try:
                    # Try to convert to numeric
                    pd.to_numeric(table.iloc[:, i], errors='raise')
                    if value_col_idx is None:
                        value_col_idx = i
                except (ValueError, TypeError):
                    # This column is not numeric, likely gene identifiers
                    if gene_col_idx is None:
                        gene_col_idx = i
        
        # Default fallback: assume first column is genes, second is values
        if gene_col_idx is None:
            gene_col_idx = 0
        if value_col_idx is None and n_cols > 1:
            value_col_idx = 1
            
        # Determine table type
        if n_cols == 2:
            table_type = "microarray"
        elif n_cols > 2:
            table_type = "rnaseq_multi_col"
        else:
            table_type = "single_col"
            
        return gene_col_idx, value_col_idx, table_type

    @staticmethod
    def load_supplementary_matrix(file_path: Union[str, Path], sep: str = ",") -> pd.DataFrame:
        """
        Load expression matrix from supplementary file (CSV/TSV, gzipped or not).
        
        Args:
            file_path: Path to the supplementary file
            sep: Separator character (comma for CSV, tab for TSV)
        
        Returns:
            DataFrame with genes as rows and samples as columns
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Supplementary file not found: {file_path}")
        
        print(f"Loading supplementary matrix from: {file_path}")
        
        # Determine if file is gzipped
        if file_path.suffix == '.gz':
            opener = gzip.open
            encoding = 'utf-8'
        else:
            opener = open
            encoding = None
        
        try:
            # Try to read the file
            with opener(file_path, 'rt', encoding=encoding) as f:
                # Read first few lines to determine structure
                first_lines = []
                for i, line in enumerate(f):
                    first_lines.append(line.strip())
                    if i >= 5:  # Read first 6 lines
                        break
            
            # Analyze structure
            print("File structure analysis:")
            for i, line in enumerate(first_lines):
                parts = line.split(sep)
                print(f"  Line {i}: {len(parts)} columns - {parts[:3]}...")
            
            # Load the full matrix
            df = pd.read_csv(file_path, sep=sep, index_col=0, compression='gzip' if file_path.suffix == '.gz' else None)
            
            print(f"Loaded matrix: {df.shape[0]} genes × {df.shape[1]} samples")
            print(f"Index (genes): {df.index[:5].tolist()}...")
            print(f"Columns (samples): {df.columns[:5].tolist()}...")
            
            return df
            
        except Exception as e:
            # Try different separators if comma doesn't work
            if sep == ",":
                print(f"Failed with comma separator, trying tab...")
                return DifferentialExpressionDataDownloader.load_supplementary_matrix(file_path, sep="\t")
            else:
                raise Exception(f"Failed to load supplementary matrix: {str(e)}")

    @staticmethod
    def _create_sample_mapping(gse, verbose=True):
        """
        Create mapping between column names in supplementary files and GSM IDs.
        
        Returns:
            Dictionary mapping various sample identifiers to GSM IDs
        """
        sample_mapping = {}
        
        if verbose:
            print("Creating sample mapping...")
        
        for gsm_id, gsm in gse.gsms.items():
            # Get sample metadata
            metadata = gsm.metadata
            
            # Common fields that might match column names
            sample_title = metadata.get('title', [''])[0]
            sample_desc = metadata.get('description', [''])[0]
            source_name = metadata.get('source_name_ch1', [''])[0]
            
            # Extract potential identifiers
            identifiers = [gsm_id, sample_title, source_name]
            
            # Look for numeric patterns (v1, v2, etc.)
            # Extract numbers from GSM ID
            gsm_number = gsm_id.replace('GSM', '')
            identifiers.extend([f"v{gsm_number}", f"V{gsm_number}"])
            
            # Look for sample order patterns
            # Sometimes files use sequential numbering
            sample_order = list(gse.gsms.keys()).index(gsm_id) + 1
            identifiers.extend([f"v{sample_order}", f"V{sample_order}", f"sample_{sample_order}"])
            
            # Add all identifiers to mapping
            for identifier in identifiers:
                if identifier and identifier.strip():
                    sample_mapping[identifier.strip()] = gsm_id
            
            if verbose:
                print(f"  {gsm_id}: {sample_title[:50]}... -> identifiers: {identifiers[:3]}")
        
        return sample_mapping

    @staticmethod
    def _map_columns_to_gsm(expr_df, gse, verbose=True):
        """
        Map expression matrix columns to GSM IDs using various strategies.
        
        Args:
            expr_df: Expression DataFrame with potentially unmapped column names
            gse: GEOparse GSE object
            verbose: Print mapping information
        
        Returns:
            DataFrame with columns renamed to GSM IDs
        """
        if verbose:
            print(f"Mapping {len(expr_df.columns)} columns to GSM IDs...")
            print(f"Original columns: {list(expr_df.columns[:5])}...")
        
        # Create sample mapping
        sample_mapping = DifferentialExpressionDataDownloader._create_sample_mapping(gse, verbose)
        
        # Strategy 1: Direct mapping using sample_mapping
        mapped_columns = {}
        for col in expr_df.columns:
            if col in sample_mapping:
                mapped_columns[col] = sample_mapping[col]
                continue
        
        if verbose and mapped_columns:
            print(f"✓ Direct mapping found {len(mapped_columns)} matches")
        
        # Strategy 2: If columns are like v1, v2, v3... map to GSM order
        if not mapped_columns and all(col.startswith(('v', 'V')) and col[1:].isdigit() for col in expr_df.columns):
            gsm_list = list(gse.gsms.keys())
            for i, col in enumerate(expr_df.columns):
                if i < len(gsm_list):
                    mapped_columns[col] = gsm_list[i]
            
            if verbose:
                print(f"✓ Sequential mapping (v1->GSM1, v2->GSM2...) found {len(mapped_columns)} matches")
        
        # Strategy 3: If columns are numeric, map to GSM order
        if not mapped_columns:
            try:
                # Check if columns can be converted to integers
                col_numbers = [int(col) for col in expr_df.columns]
                # Sort by number to get correct order
                col_order = sorted(zip(col_numbers, expr_df.columns))
                gsm_list = list(gse.gsms.keys())
                
                for i, (_, col) in enumerate(col_order):
                    if i < len(gsm_list):
                        mapped_columns[col] = gsm_list[i]
                
                if verbose:
                    print(f"✓ Numeric order mapping found {len(mapped_columns)} matches")
            except ValueError:
                pass
        
        # Strategy 4: Try partial string matching with sample titles
        if len(mapped_columns) < len(expr_df.columns):
            remaining_cols = [col for col in expr_df.columns if col not in mapped_columns]
            
            for col in remaining_cols:
                best_match = None
                best_score = 0
                
                for gsm_id, gsm in gse.gsms.items():
                    if gsm_id in mapped_columns.values():
                        continue  # Already mapped
                    
                    sample_title = gsm.metadata.get('title', [''])[0].lower()
                    col_lower = str(col).lower()
                    
                    # Simple string similarity
                    if col_lower in sample_title or sample_title in col_lower:
                        score = min(len(col_lower), len(sample_title)) / max(len(col_lower), len(sample_title))
                        if score > best_score:
                            best_score = score
                            best_match = gsm_id
                
                if best_match and best_score > 0.3:  # Arbitrary threshold
                    mapped_columns[col] = best_match
            
            if verbose and len(mapped_columns) > len(expr_df.columns) - len(remaining_cols):
                print(f"✓ String matching found {len(mapped_columns) - (len(expr_df.columns) - len(remaining_cols))} additional matches")
        
        # Apply mapping
        if mapped_columns:
            expr_df_mapped = expr_df.rename(columns=mapped_columns)
            
            if verbose:
                print(f"  Mapping Results:")
                print(f"  Total columns: {len(expr_df.columns)}")
                print(f"  Successfully mapped: {len(mapped_columns)}")
                print(f"  Unmapped: {len(expr_df.columns) - len(mapped_columns)}")
                
                if len(mapped_columns) < len(expr_df.columns):
                    unmapped = [col for col in expr_df.columns if col not in mapped_columns]
                    print(f"  Unmapped columns: {unmapped[:5]}...")
                
                # Show some mapping examples
                mapping_examples = list(mapped_columns.items())[:3]
                for orig, gsm in mapping_examples:
                    print(f"  {orig} -> {gsm}")
                if len(mapped_columns) > 3:
                    print(f"  ... and {len(mapped_columns) - 3} more")
            
            return expr_df_mapped
        else:
            if verbose:
                print("   No column mapping found - returning original DataFrame")
                print("   This might cause issues with sample matching!")
            return expr_df

    @staticmethod
    def get_expression_matrix(gse, verbose=True, try_supplementary=True, dest_dir="./geo_data"):
        """
        Constructs a gene expression matrix from GSM tables OR supplementary files.
        This method tries multiple approaches:
        1. Extract from GSM tables (works for microarray)
        2. Download and use supplementary files (works for RNA-seq)
        
        Args:
            gse: GEOparse GSE object
            verbose: Print detailed information
            try_supplementary: Whether to try supplementary files if GSM tables fail
            dest_dir: Directory to save/look for supplementary files
        
        Returns:
            DataFrame with genes as rows and GSM IDs as columns
        """
        
        print("=" * 60)
        print(f"EXTRACTING EXPRESSION MATRIX FOR {gse.name}")
        print("=" * 60)
        
        # First, try the traditional GSM table approach
        try:
            if verbose:
                print("APPROACH 1: Extracting from GSM tables...")
            
            gene_expr = {}
            skipped_samples = []
            table_structures = {}
            empty_tables = 0

            if verbose:
                print(f"Processing {len(gse.gsms)} samples...")

            for gsm_id, gsm in gse.gsms.items():
                table = gsm.table
                
                # Detect table structure
                gene_col_idx, value_col_idx, table_type = DifferentialExpressionDataDownloader._detect_table_structure(table)
                
                if table_type == "empty":
                    empty_tables += 1
                    skipped_samples.append((gsm_id, "Empty table"))
                    continue
                    
                if gene_col_idx is None or value_col_idx is None:
                    skipped_samples.append((gsm_id, "Could not identify gene/value columns"))
                    continue
                    
                table_structures[gsm_id] = table_type
                
                try:
                    # Extract gene identifiers and values
                    genes = table.iloc[:, gene_col_idx].astype(str)
                    values = pd.to_numeric(table.iloc[:, value_col_idx], errors="coerce")
                    
                    # Filter out NaN values and empty gene identifiers
                    valid_mask = ~values.isna() & (genes != '') & (genes != 'nan')
                    
                    if valid_mask.sum() == 0:
                        skipped_samples.append((gsm_id, "No valid gene-value pairs"))
                        continue
                    
                    genes_filtered = genes[valid_mask]
                    values_filtered = values[valid_mask]
                    
                    # Create series with gene identifiers as index
                    gene_expr[gsm_id] = pd.Series(values_filtered.values, index=genes_filtered)
                    
                    if verbose:
                        print(f"✓ {gsm_id}: {len(genes_filtered)} genes, table type: {table_type}")
                        
                except Exception as e:
                    skipped_samples.append((gsm_id, f"Parsing error: {str(e)}"))
                    if verbose:
                        print(f"✗ {gsm_id}: {str(e)}")
                    continue

            # Check if GSM approach worked
            if gene_expr:
                if verbose:
                    print(f"SUCCESS: GSM tables approach worked!")
                    print(f"Successfully processed {len(gene_expr)} samples")
                    if table_structures:
                        print(f"Table types found: {set(table_structures.values())}")

                # Create expression matrix
                expr_df = pd.DataFrame(gene_expr)
                expr_df = expr_df.dropna(how='all')
                
                if verbose:
                    print(f"Final expression matrix: {expr_df.shape[0]} genes × {expr_df.shape[1]} samples")
                    print(f"Missing values: {expr_df.isna().sum().sum()}")

                return expr_df
            
            else:
                if verbose:
                    print(f"GSM tables approach failed:")
                    print(f"  - Empty tables: {empty_tables}/{len(gse.gsms)}")
                    print(f"  - Skipped samples: {len(skipped_samples)}")
                    if skipped_samples:
                        for gsm_id, reason in skipped_samples[:3]:
                            print(f"    * {gsm_id}: {reason}")
                        if len(skipped_samples) > 3:
                            print(f"    * ... and {len(skipped_samples) - 3} more")
                
                if not try_supplementary:
                    raise ValueError("No valid expression tables found in GSM objects and supplementary file approach disabled.")
        
        except Exception as e:
            if not try_supplementary:
                raise e
            if verbose:
                print(f"GSM tables approach failed: {str(e)}")
        
        # Try supplementary files approach
        if try_supplementary:
            if verbose:
                print(f"APPROACH 2: Trying supplementary files...")
            
            try:
                # Download supplementary files
                downloaded_files = DifferentialExpressionDataDownloader.download_supplementary_files(
                    gse.name, 
                    dest_dir=dest_dir,
                    file_patterns=['*count*', '*matrix*', '*fpkm*', '*tpm*', '*expression*']
                )
                
                if not downloaded_files:
                    raise ValueError("No supplementary files found")
                
                # Try to load each downloaded file
                for file_path in downloaded_files:
                    try:
                        if verbose:
                            print(f"Trying to load: {file_path.name}")
                        
                        expr_df = DifferentialExpressionDataDownloader.load_supplementary_matrix(file_path)
                        
                        # Map columns to GSM IDs
                        expr_df_mapped = DifferentialExpressionDataDownloader._map_columns_to_gsm(
                            expr_df, gse, verbose=verbose
                        )
                        
                        if verbose:
                            print(f"SUCCESS: Loaded matrix from supplementary file!")
                            print(f"Matrix shape: {expr_df_mapped.shape[0]} genes × {expr_df_mapped.shape[1]} samples")
                            print(f"Mapped sample names: {list(expr_df_mapped.columns[:5])}...")
                            print(f"Gene names: {list(expr_df_mapped.index[:5])}...")
                        
                        return expr_df_mapped
                        
                    except Exception as e:
                        if verbose:
                            print(f"Failed to load {file_path.name}: {str(e)}")
                        continue
                
                raise ValueError("Could not load any supplementary files")
                
            except Exception as e:
                if verbose:
                    print(f"Supplementary files approach failed: {str(e)}")
                raise ValueError(f"Both GSM tables and supplementary files approaches failed. "
                               f"GSM issue: empty tables. Supplementary issue: {str(e)}")
        
        # If we get here, both approaches failed
        raise ValueError("No valid expression data found using any available method.")

    @staticmethod
    def diagnose_gse_structure(gse, max_samples_to_check=5):
        """
        Diagnostic function to understand the structure of GSM tables in a GSE object.
        Useful for debugging RNA-seq vs microarray format issues.
        """
        print(f"=== GSE Structure Diagnosis for {gse.name} ===")
        print(f"Total samples: {len(gse.gsms)}")
        print(f"Total platforms: {len(gse.gpls)}")
        
        # Check platform information
        for gpl_id, gpl in gse.gpls.items():
            print(f"\nPlatform {gpl_id}:")
            print(f"  Title: {gpl.metadata.get('title', ['Unknown'])[0]}")
            print(f"  Technology: {gpl.metadata.get('technology', ['Unknown'])[0]}")
            print(f"  Organism: {gpl.metadata.get('taxid', ['Unknown'])[0]}")
        
        # Sample a few GSMs to understand table structure
        sample_gsms = list(gse.gsms.items())[:max_samples_to_check]
        
        print(f"\n=== Sample Analysis (first {len(sample_gsms)} samples) ===")
        
        empty_count = 0
        for gsm_id, gsm in sample_gsms:
            print(f"\nSample {gsm_id}:")
            print(f"  Platform: {gsm.metadata.get('platform_id', ['Unknown'])[0]}")
            
            table = gsm.table
            if table is None:
                print("  Table: None")
                empty_count += 1
                continue
            elif table.empty:
                print("  Table: Empty")
                empty_count += 1
                continue
                
            print(f"  Table shape: {table.shape}")
            print(f"  Columns: {list(table.columns)}")
            
            # Show first few rows
            print("  First 3 rows:")
            for i in range(min(3, len(table))):
                row_data = table.iloc[i].tolist()
                print(f"    Row {i}: {row_data}")
                
            # Check data types
            print("  Column data types:")
            for col in table.columns:
                dtype = table[col].dtype
                non_null_count = table[col].count()
                print(f"    {col}: {dtype} ({non_null_count} non-null values)")
        
        # Summary
        print(f"\n=== SUMMARY ===")
        print(f"Empty tables: {empty_count}/{len(sample_gsms)} samples checked")
        if empty_count == len(sample_gsms):
            print("   ALL TABLES ARE EMPTY - This is typical for RNA-seq data!")
            print("   RNA-seq data is usually provided as supplementary files.")
            print("   Use get_expression_matrix() with try_supplementary=True.")
        
        print("\n" + "="*50)