from .de_filter_downloader import DifferentialExpressionDataDownloader
from .de_filter import PhenotypeInferencer
from .de_platform_inspector import PlatformInspector
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind

class DEScorer:
    def __init__(self, gse_id: str, built_in_phenotype_inferencer: bool = True, max_workers: int = 5):
        self.gse_id = gse_id
        self.max_workers = max_workers
        self._loader_class = DifferentialExpressionDataDownloader()
        self.gse_data = self._loader_class.fetch_from_geo(self.gse_id)
        self.expression_matrix = self._loader_class.get_expression_matrix(self.gse_data)
        self.phenotypes = PhenotypeInferencer(self.max_workers)
