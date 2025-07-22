import requests
from typing import List, Dict, Optional, Literal
import pandas as pd

class OpenTargetsFetcher:

    def __init__(self, disease_name: str, return_dataframe: bool, sort: Literal["asc", "desc"]):
        self.disease_name = disease_name
        self._GQL_URL = "https://api.platform.opentargets.org/api/v4/graphql"
        self.return_dataframe = return_dataframe
        self.sort = sort
        self._targets: Optional[List[Dict]] = None
    
    def _get_disease_id(self) -> str:
        query = f"""
        {{
            search(queryString: "{self.disease_name}") {{
                hits {{
                id
                name
                entity
                }}
            }}
        }}
        """
        try:
            response = requests.post(self._GQL_URL, json={"query":query})
            return response.json()['data']['search']['hits'][0]['id']
        except Exception as e:
            raise Exception(f"Error at finding disease id for: {self.disease_name}, {str(e)}")

    def _get_targets_disease(self) -> None:
        query = f"""
        query {{
            disease(efoId: "{self._get_disease_id()}") {{
                associatedTargets(page: {{ index: 0, size: 100 }}) {{
                count
                rows {{
                    target {{
                            id
                            approvedSymbol
                            approvedName
                        }}
                        score
                    }}
                }}
            }}
        }}
        """
        try:
            response = requests.post(self._GQL_URL, json={"query":query})
            self._targets = [
                { 
                    "gene_id": i['target']['id'],
                    "symbol":  i['target']['approvedSymbol'],
                    "name": i['target']['approvedName'],
                    "score": i['score']
                }

                for i in response.json()['data']['disease']['associatedTargets']['rows']
            ]
        except Exception as e:
            raise Exception(f"Error fetching targets for disease: {self.disease_name}, {str(e)}")
        
    def get_target_genes(self) -> List[Dict] | pd.DataFrame: 
        if self._targets is None:
            self._get_targets_disease()
        genes = self._targets
        if self.return_dataframe:
            return pd.DataFrame(genes).sort_values(by="score", ascending=(self.sort == "asc"))
        return sorted(genes, key=lambda x: x["score"], reverse=(self.sort == "desc"))