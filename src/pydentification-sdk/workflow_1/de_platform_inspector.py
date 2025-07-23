import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class PlatformResult:
    """Result object containing platform prediction and confidence metrics"""
    platform_type: str
    confidence: float
    method: str  # 'metadata', 'data_analysis', or 'combined'
    details: Dict

class PlatformInspector:
    def __init__(self, gse_obj):
        self.gse = gse_obj
        self.platforms = list(gse_obj.gpls.values())
        self.logger = logging.getLogger(__name__)
        
        # Enhanced keyword sets
        self.known_rnaseq_keywords = {
            "RNA-Seq", "HiSeq", "NextSeq", "NovaSeq", "MiSeq", "Ion Torrent",
            "PacBio", "Oxford Nanopore", "single-cell", "scRNA-seq", "bulk RNA-seq",
            "RNA sequencing", "transcriptome sequencing", "Illumina HiSeq",
            "HiSeq 2000", "HiSeq 2500", "HiSeq 4000", "Genome Analyzer"
        }
        
        self.known_microarray_keywords = {
            "Affymetrix", "GeneChip", "Illumina array", "BeadArray", "BeadChip",
            "Human Genome U133", "Mouse Genome 430", "HumanHT-12", "MouseWG-6",
            "Agilent", "CodeLink", "Sentrix", "microarray", "gene expression array"
        }
        
        self.known_methylation_keywords = {
            "methylation", "450K", "850K", "EPIC", "Infinium", "bisulfite",
            "HumanMethylation450", "HumanMethylationEPIC", "CpG"
        }
        
        self.known_snp_keywords = {
            "SNP", "genotyping", "genome-wide association", "GWAS", "SNP6.0",
            "CytoSNP", "OmniExpress", "HumanOmni"
        }
        
        self.known_proteomics_keywords = {
            "proteomics", "mass spectrometry", "LC-MS", "protein", "peptide",
            "iTRAQ", "TMT", "SILAC"
        }

    def detect_platform_type(self) -> PlatformResult:
        """
        Detect platform type using multiple approaches with confidence scoring
        """
        # Try metadata-based detection first
        metadata_result = self._detect_by_metadata()
        
        if metadata_result.confidence > 0.8:
            self.logger.info(f"High confidence metadata detection: {metadata_result.platform_type}")
            return metadata_result
        
        # Fallback to data distribution analysis
        data_result = self._detect_by_data_distribution()
        
        # Combine results if both have moderate confidence
        if metadata_result.confidence > 0.3 and data_result.confidence > 0.3:
            combined_result = self._combine_results(metadata_result, data_result)
            return combined_result
        
        # Return the result with higher confidence
        if metadata_result.confidence >= data_result.confidence:
            return metadata_result
        else:
            return data_result

    def _detect_by_metadata(self) -> PlatformResult:
        """Enhanced metadata-based platform detection with confidence scoring"""
        platform_scores = {
            "rnaseq": 0.0,
            "microarray": 0.0,
            "methylation": 0.0,
            "snp": 0.0,
            "proteomics": 0.0
        }
        
        total_platforms = len(self.platforms)
        if total_platforms == 0:
            return PlatformResult("unknown", 0.0, "metadata", {"reason": "No platforms found"})
        
        keyword_matches = []
        
        for gpl in self.platforms:
            try:
                title = gpl.metadata.get("title", [""])[0].lower()
                technology = gpl.metadata.get("technology", [""])[0].lower()
                summary = gpl.metadata.get("summary", [""])[0].lower()
                full_info = f"{title} {technology} {summary}"
                
                # Check for each platform type
                rnaseq_matches = sum(1 for kw in self.known_rnaseq_keywords if kw.lower() in full_info)
                microarray_matches = sum(1 for kw in self.known_microarray_keywords if kw.lower() in full_info)
                methylation_matches = sum(1 for kw in self.known_methylation_keywords if kw.lower() in full_info)
                snp_matches = sum(1 for kw in self.known_snp_keywords if kw.lower() in full_info)
                proteomics_matches = sum(1 for kw in self.known_proteomics_keywords if kw.lower() in full_info)
                
                platform_scores["rnaseq"] += rnaseq_matches
                platform_scores["microarray"] += microarray_matches
                platform_scores["methylation"] += methylation_matches
                platform_scores["snp"] += snp_matches
                platform_scores["proteomics"] += proteomics_matches
                
                keyword_matches.append({
                    "gpl_id": getattr(gpl, 'name', 'unknown'),
                    "matches": {
                        "rnaseq": rnaseq_matches,
                        "microarray": microarray_matches,
                        "methylation": methylation_matches,
                        "snp": snp_matches,
                        "proteomics": proteomics_matches
                    }
                })
                
            except Exception as e:
                self.logger.warning(f"Error processing platform metadata: {e}")
                continue
        
        # Determine best match
        if max(platform_scores.values()) == 0:
            return PlatformResult("unknown", 0.0, "metadata", {
                "reason": "No keyword matches found",
                "keyword_matches": keyword_matches
            })
        
        best_platform = max(platform_scores.keys(), key=lambda k: platform_scores[k])
        max_score = platform_scores[best_platform]
        
        # Calculate confidence based on score strength and consistency
        total_matches = sum(platform_scores.values())
        confidence = (max_score / total_matches) * min(1.0, total_matches / total_platforms)
        
        return PlatformResult(
            platform_type=best_platform,
            confidence=confidence,
            method="metadata",
            details={
                "platform_scores": platform_scores,
                "keyword_matches": keyword_matches,
                "total_platforms": total_platforms
            }
        )

    def _detect_by_data_distribution(self, min_samples=3, max_samples=10, 
                                   values_per_sample=200) -> PlatformResult:
        """Enhanced data distribution analysis with multiple heuristics"""
        
        sample_data = self._get_representative_sample(min_samples, max_samples, values_per_sample)
        
        if not sample_data:
            return PlatformResult("unknown", 0.0, "data_analysis", 
                                {"reason": "No valid sample data found"})
        
        all_values = []
        sample_stats = []
        
        for sample_id, values in sample_data.items():
            if len(values) > 0:
                all_values.extend(values)
                sample_stats.append({
                    "sample_id": sample_id,
                    "mean": np.mean(values),
                    "median": np.median(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "int_ratio": np.sum(np.array(values) % 1 == 0) / len(values),
                    "negative_ratio": np.sum(np.array(values) < 0) / len(values)
                })
        
        if len(all_values) == 0:
            return PlatformResult("unknown", 0.0, "data_analysis", 
                                {"reason": "No numeric values found"})
        
        # Calculate aggregate statistics
        all_values = np.array(all_values)
        
        # Remove infinite and NaN values
        all_values = all_values[np.isfinite(all_values)]
        
        if len(all_values) == 0:
            return PlatformResult("unknown", 0.0, "data_analysis", 
                                {"reason": "No finite values found"})
        
        stats = {
            "total_values": len(all_values),
            "mean": np.mean(all_values),
            "median": np.median(all_values),
            "std": np.std(all_values),
            "min": np.min(all_values),
            "max": np.max(all_values),
            "int_ratio": np.sum(all_values % 1 == 0) / len(all_values),
            "negative_ratio": np.sum(all_values < 0) / len(all_values),
            "zero_ratio": np.sum(all_values == 0) / len(all_values),
            "dynamic_range": np.log10(np.max(all_values) + 1) - np.log10(np.min(all_values) + 1) if np.min(all_values) >= 0 else 0,
            "cv": np.std(all_values) / np.mean(all_values) if np.mean(all_values) != 0 else 0
        }
        
        # Enhanced heuristics for platform detection
        platform_scores = self._calculate_platform_scores(stats)
        
        best_platform = max(platform_scores.keys(), key=lambda k: platform_scores[k])
        confidence = platform_scores[best_platform]
        
        return PlatformResult(
            platform_type=best_platform,
            confidence=confidence,
            method="data_analysis",
            details={
                "statistics": stats,
                "platform_scores": platform_scores,
                "sample_stats": sample_stats,
                "samples_analyzed": len(sample_stats)
            }
        )

    def _get_representative_sample(self, min_samples: int, max_samples: int, 
                                 values_per_sample: int) -> Dict[str, List[float]]:
        """Get representative sample data from multiple GSMs"""
        sample_data = {}
        processed_samples = 0
        
        # Get list of GSM IDs and sample evenly
        gsm_ids = list(self.gse.gsms.keys())
        if len(gsm_ids) > max_samples:
            # Sample evenly across the dataset
            step = len(gsm_ids) // max_samples
            gsm_ids = gsm_ids[::step][:max_samples]
        
        for gsm_id in gsm_ids:
            if processed_samples >= max_samples:
                break
                
            try:
                gsm = self.gse.gsms[gsm_id]
                values = self._extract_numeric_values(gsm, values_per_sample)
                
                if values:  # Only include if we got valid data
                    sample_data[gsm_id] = values
                    processed_samples += 1
                    
            except Exception as e:
                self.logger.warning(f"Failed to process sample {gsm_id}: {e}")
                continue
        
        if processed_samples < min_samples:
            self.logger.warning(f"Only processed {processed_samples} samples, minimum was {min_samples}")
        
        return sample_data

    def _extract_numeric_values(self, gsm, limit: int) -> List[float]:
        """Extract numeric values from a GSM, trying multiple columns"""
        values = []
        
        try:
            table = gsm.table
            if table is None or table.empty:
                return values
            
            # Try multiple columns (skip first column which is usually IDs)
            for col_idx in range(1, min(table.shape[1], 4)):  # Check up to 3 data columns
                try:
                    col_values = table.iloc[:limit, col_idx].astype(str).str.replace(",", "")
                    # Try to convert to float
                    numeric_values = pd.to_numeric(col_values, errors='coerce')
                    # Remove NaN values
                    valid_values = numeric_values.dropna().tolist()
                    
                    if len(valid_values) > len(values):
                        values = valid_values
                        
                except (ValueError, AttributeError, KeyError) as e:
                    continue
            
        except Exception as e:
            self.logger.warning(f"Error extracting values from GSM: {e}")
        
        return values[:limit]

    def _calculate_platform_scores(self, stats: Dict) -> Dict[str, float]:
        """Calculate platform likelihood scores based on data statistics"""
        scores = {
            "rnaseq": 0.0,
            "microarray": 0.0,
            "methylation": 0.0,
            "proteomics": 0.0,
            "unknown": 0.1
        }
        
        # RNA-seq indicators
        if stats["int_ratio"] > 0.9 and stats["median"] > 5:
            scores["rnaseq"] += 0.4
        if stats["dynamic_range"] > 3:  # More than 3 orders of magnitude
            scores["rnaseq"] += 0.3
        if stats["zero_ratio"] > 0.1:  # Many zeros (typical in RNA-seq)
            scores["rnaseq"] += 0.2
        if stats["cv"] > 1.0:  # High coefficient of variation
            scores["rnaseq"] += 0.2
        if stats["negative_ratio"] < 0.01:  # Very few negative values
            scores["rnaseq"] += 0.1
        
        # Microarray indicators
        if stats["int_ratio"] < 0.1 and stats["negative_ratio"] < 0.05:
            scores["microarray"] += 0.4
        if 1 < stats["median"] < 20:  # Typical microarray range
            scores["microarray"] += 0.3
        if 0.5 < stats["cv"] < 2.0:  # Moderate coefficient of variation
            scores["microarray"] += 0.2
        if stats["dynamic_range"] < 2:  # Smaller dynamic range
            scores["microarray"] += 0.2
        
        # Methylation array indicators (beta values between 0 and 1)
        if stats["min"] >= 0 and stats["max"] <= 1.1 and stats["median"] < 1:
            scores["methylation"] += 0.6
        if 0.2 < stats["median"] < 0.8:
            scores["methylation"] += 0.3
        
        # Proteomics indicators
        if stats["dynamic_range"] > 4 and stats["cv"] > 2.0:
            scores["proteomics"] += 0.4
        if stats["negative_ratio"] > 0.05:  # Can have negative values (log ratios)
            scores["proteomics"] += 0.2
        
        # Normalize scores
        max_score = max(scores.values())
        if max_score > 0:
            for platform in scores:
                scores[platform] = min(1.0, scores[platform] / max_score)
        
        return scores

    def _combine_results(self, metadata_result: PlatformResult, 
                        data_result: PlatformResult) -> PlatformResult:
        """Combine metadata and data analysis results"""
        
        # If both methods agree, increase confidence
        if metadata_result.platform_type == data_result.platform_type:
            combined_confidence = min(1.0, 
                (metadata_result.confidence + data_result.confidence) * 0.7)
            return PlatformResult(
                platform_type=metadata_result.platform_type,
                confidence=combined_confidence,
                method="combined",
                details={
                    "metadata_result": metadata_result,
                    "data_result": data_result,
                    "agreement": True
                }
            )
        
        # If they disagree, use weighted average based on confidence
        if metadata_result.confidence > data_result.confidence:
            primary_result = metadata_result
            secondary_result = data_result
        else:
            primary_result = data_result
            secondary_result = metadata_result
        
        weight_primary = primary_result.confidence / (primary_result.confidence + secondary_result.confidence)
        combined_confidence = (primary_result.confidence * weight_primary + 
                             secondary_result.confidence * (1 - weight_primary)) * 0.6
        
        return PlatformResult(
            platform_type=primary_result.platform_type,
            confidence=combined_confidence,
            method="combined",
            details={
                "metadata_result": metadata_result,
                "data_result": data_result,
                "agreement": False,
                "primary_method": primary_result.method,
                "secondary_method": secondary_result.method,
                "confidence_weights": {
                    "primary": weight_primary,
                    "secondary": 1 - weight_primary
                }
            }
        )