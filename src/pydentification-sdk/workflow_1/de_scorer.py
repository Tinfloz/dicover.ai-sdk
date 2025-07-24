from .de_filter_downloader import DifferentialExpressionDataDownloader
from .de_filter import PhenotypeInferencer
from .de_platform_inspector import PlatformInspector
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from typing import Optional, Dict
import logging
import mygene
from statsmodels.stats.multitest import multipletests

mg_client = mygene.MyGeneInfo()


class DEScorer:
    def __init__(
        self,
        phenotypes: Optional[Dict[str, str]],
        gse_id: str,
        built_in_phenotype_inferencer: bool = True,
        max_workers: int = 5,
        w_fc: float = 1.0,
        w_fdr: float = 1.0
    ):
        self.gse_id = gse_id
        self.phenotypes = phenotypes
        self.max_workers = max_workers
        self.w_fc = w_fc
        self.w_fdr = w_fdr
        self._logger = logging.getLogger(__name__)
        self._loader_class = DifferentialExpressionDataDownloader()
        self._gse_data = self._loader_class.fetch_from_geo(self.gse_id)

        self._inferred_phenotypes = (
            dict(PhenotypeInferencer(self.max_workers).run(self._gse_data.gsms))
            if built_in_phenotype_inferencer
            else self.phenotypes
        )

        self._platform = PlatformInspector(self._gse_data).detect_platform_type()
        raw_expression = self._loader_class.get_expression_matrix(self._gse_data)
        self._expression_matrix = self._map_to_gene_level(self._gse_data, raw_expression)

        self._logger.info("Splitting samples between CONTROL and DISEASED...")
        self._subset_expression_matrix()
        self._logger.info("Control samples: %d | Disease samples: %d",
                          len(self._control_samples), len(self._diseased_samples))

    def _subset_expression_matrix(self) -> None:
        valid_samples = [
            sample for sample in list(self._inferred_phenotypes.keys())
            if sample in self._expression_matrix.columns
        ]
        if len(valid_samples) < 4:
            raise ValueError("Too few labeled samples found in expression matrix.")

        self._expression_matrix = self._expression_matrix.loc[:, valid_samples]
        self._control_samples = [s for s in valid_samples if self._inferred_phenotypes[s].upper() == "CONTROL"]
        self._diseased_samples = [s for s in valid_samples if self._inferred_phenotypes[s].upper() == "DISEASED"]

        if len(self._control_samples) < 2 or len(self._diseased_samples) < 2:
            raise ValueError("Need at least 2 samples in both CONTROL and DISEASED groups.")

    def _map_to_gene_level(self, gse, expression_matrix: pd.DataFrame) -> pd.DataFrame:
        if self._platform.platform_type == "rnaseq":
            if expression_matrix.index.str.startswith("ENSG").any():
                self._logger.info("RNA-seq: Mapping Ensembl IDs to gene symbols...")
                return DEScorer.convert_ensembl_to_symbol(expression_matrix)
            else:
                self._logger.info("RNA-seq: Gene names already mapped.")
                return expression_matrix
        else:
            gpl = list(gse.gpls.values())[0]
            platform_table = gpl.table

            possible_cols = ["Gene Symbol", "gene_symbol", "GENE_SYMBOL", "Symbol"]
            symbol_col = next((col for col in possible_cols if col in platform_table.columns), None)

            if symbol_col is None:
                raise ValueError("Gene symbol column not found in platform metadata.")

            mapping_df = platform_table.reset_index()[["ID", symbol_col]].dropna()
            mapping_df.columns = ["probe_id", "gene"]

            # Split multi-mapped gene symbols and clean
            mapping_df["gene"] = mapping_df["gene"].str.split(r" /// |;|,|\|")
            mapping_df = mapping_df.explode("gene")
            mapping_df["gene"] = mapping_df["gene"].str.strip()
            mapping_df = mapping_df[mapping_df["gene"] != ""].dropna().set_index("probe_id")

            expression_matrix = expression_matrix.copy()
            mapping_dict = mapping_df["gene"].to_dict()
            expression_matrix["gene"] = expression_matrix.index.map(mapping_dict)
            expression_matrix = expression_matrix.dropna(subset=["gene"])
            expression_matrix = expression_matrix.groupby("gene").mean()
            return expression_matrix

    @staticmethod
    def convert_ensembl_to_symbol(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        ensembl_ids = df.index.tolist()
        query = mg_client.querymany(
            ensembl_ids,
            scopes="ensembl.gene",
            fields="symbol",
            species="human",
            as_dataframe=True,
            returnall=False,
            verbose=False,
        )
        symbol_map = query["symbol"].dropna().to_dict()
        df["gene"] = df.index.map(symbol_map)
        df = df.dropna(subset=["gene"])
        df = df.groupby("gene").mean()
        return df

    def score_genes(self) -> pd.DataFrame:
        self._logger.info("Scoring genes via differential expression...")
        control = self._expression_matrix[self._control_samples]
        disease = self._expression_matrix[self._diseased_samples]
        genes = self._expression_matrix.index

        log2fc_list = []
        pval_list = []

        for gene in genes:
            ctrl_vals = control.loc[gene].dropna()
            dis_vals = disease.loc[gene].dropna()

            if len(ctrl_vals) < 2 or len(dis_vals) < 2:
                log2fc_list.append(np.nan)
                pval_list.append(np.nan)
                continue

            mean_ctrl = np.mean(ctrl_vals)
            mean_dis = np.mean(dis_vals)
            log2fc = np.log2(mean_dis + 1e-9) - np.log2(mean_ctrl + 1e-9)

            try:
                _, pval = ttest_ind(dis_vals, ctrl_vals, equal_var=False)
            except Exception as e:
                self._logger.warning(f"t-test failed for {gene}: {e}")
                log2fc = np.nan
                pval = np.nan

            log2fc_list.append(log2fc)
            pval_list.append(pval)

        fdr_array = multipletests(pval_list, method="fdr_bh")[1]

        result_df = pd.DataFrame({
            "gene": genes,
            "log2FC": log2fc_list,
            "pval": pval_list,
            "fdr": fdr_array
        }).dropna()

        result_df["DE_score"] = (
            self.w_fc * np.abs(result_df["log2FC"]) +
            self.w_fdr * -np.log10(result_df["fdr"] + 1e-9)
        )

        result_df = result_df.sort_values("DE_score", ascending=False).reset_index(drop=True)
        self._logger.info(f"DE scoring complete: {len(result_df)} genes scored.")
        return result_df