from typing import Literal, Optional, List, Dict
from pathlib import Path
import pandas as pd

Tissue = Literal[
    'Artery - Tibial', 'Brain - Hypothalamus', 'Liver', 'Skin - Sun Exposed (Lower leg)',
    'Cervix - Endocervix', 'Vagina', 'Brain - Cerebellar Hemisphere', 'Cervix - Ectocervix',
    'Adrenal Gland', 'Brain - Hippocampus', 'Spleen', 'Heart - Atrial Appendage',
    'Brain - Caudate (basal ganglia)', 'Brain - Substantia nigra', 'Testis', 'Lung',
    'Artery - Aorta', 'Esophagus - Mucosa', 'Stomach', 'Colon - Sigmoid',
    'Brain - Frontal Cortex (BA9)', 'Kidney - Medulla', 'Adipose - Subcutaneous', 'Pancreas',
    'Artery - Coronary', 'Muscle - Skeletal', 'Prostate', 'Skin - Not Sun Exposed (Suprapubic)',
    'Whole Blood', 'Brain - Nucleus accumbens (basal ganglia)', 'Fallopian Tube',
    'Esophagus - Muscularis', 'Pituitary', 'Ovary', 'Small Intestine - Terminal Ileum',
    'Brain - Spinal cord (cervical c-1)', 'Uterus', 'Thyroid', 'Adipose - Visceral (Omentum)',
    'Nerve - Tibial', 'Brain - Amygdala', 'Brain - Anterior cingulate cortex (BA24)',
    'Brain - Cortex', 'Cells - EBV-transformed lymphocytes', 'Colon - Transverse',
    'Breast - Mammary Tissue', 'Kidney - Cortex', 'Minor Salivary Gland',
    'Brain - Putamen (basal ganglia)', 'Cells - Cultured fibroblasts', 'Heart - Left Ventricle',
    'Esophagus - Gastroesophageal Junction', 'Bladder', 'Brain - Cerebellum',
    "Brain", "Kidney", "Colon", "Skin"
]

class GTExFilter:
    
    def __init__(self, tissue: Tissue, file_path: Optional[str] = None, return_dataframe: bool = False):
        self.tissue = tissue
        self.return_dataframe = return_dataframe

        if file_path is not None:
            self.file_path = Path(file_path)
        else:
            current_dir = Path(__file__).resolve().parent
            root_dir = current_dir.parents[2]
            self.file_path = root_dir / "data" / "gtex" / "GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct.gz"

        try:
            self.dataframe_gtex = pd.read_csv(self.file_path, sep="\t", skiprows=2, compression='gzip')
        except Exception as e:
            raise FileNotFoundError(f"Failed to load GTEx data from {self.file_path}: {str(e)}")

    def load_gtex(self) -> pd.DataFrame | List[Dict]:
        tissue_cols = [col for col in self.dataframe_gtex.columns if self.tissue in col]
        if not tissue_cols:
            raise ValueError(f"No columns found matching tissue: '{self.tissue}'. Available columns: {list(self.dataframe_gtex.columns[-10:])}")
        df = self.dataframe_gtex.copy()
        df["median_tpm"] = df[tissue_cols].median(axis=1)
        df = df.rename(columns={"Name": "gene_id", "Description": "symbol"})
        df["gene_id"] = df["gene_id"].str.split(".").str[0]
        return df[["gene_id", "symbol", "median_tpm"]] if self.return_dataframe else df[["gene_id", "symbol", "median_tpm"]].to_dict(orient="records")