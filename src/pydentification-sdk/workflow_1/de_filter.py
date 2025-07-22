import os
import time
import json
import torch
import random
import logging
import requests
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.exceptions import RequestException

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class PhenotypeInferencer:
    def __init__(
        self,
        api_url: str,
        api_key: str,
        deployment_id: str,
        api_version: str = "2024-02-15-preview",
        max_workers: int = 4,
        max_retries: int = 5,
        base_delay: float = 1.0,
        verbose: bool = True,
    ):
        self.api_url = f"{api_url}/openai/deployments/{deployment_id}/chat/completions?api-version={api_version}"
        self.headers = {
            "Content-Type": "application/json",
            "api-key": api_key
        }
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.verbose = verbose

    def _log(self, msg):
        if self.verbose:
            logging.info(msg)

    def _infer_single_sample(self, sample: Dict) -> Dict:
        """Send a single sample to the OpenAI model and return result with inference."""
        prompt = (
            "Classify the given sample into one of the two categories only: 'control' or 'diseased'.\n\n"
            f"Sample:\n{json.dumps(sample, indent=2)}\n\n"
            "Response should be only one word: 'control' or 'diseased'."
        )

        body = {
            "messages": [
                {"role": "system", "content": "You are a biomedical AI model trained to classify samples."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0,
            "top_p": 1,
            "n": 1
        }

        for attempt in range(self.max_retries):
            try:
                response = requests.post(self.api_url, headers=self.headers, json=body)
                if response.status_code == 200:
                    reply = response.json()["choices"][0]["message"]["content"].strip().lower()
                    sample["phenotype_inference"] = reply
                    return sample
                elif response.status_code == 429:
                    delay = self.base_delay * (2 ** attempt) + random.uniform(0.1, 0.5)
                    self._log(f"Rate limited (429). Retry {attempt + 1}/{self.max_retries} in {delay:.2f}s.")
                    time.sleep(delay)
                elif response.status_code >= 500:
                    delay = self.base_delay * (2 ** attempt) + random.uniform(0.1, 0.5)
                    self._log(f"Server error {response.status_code}. Retry {attempt + 1}/{self.max_retries} in {delay:.2f}s.")
                    time.sleep(delay)
                else:
                    self._log(f"Request failed with status {response.status_code}: {response.text}")
                    sample["phenotype_inference"] = "error"
                    return sample
            except RequestException as e:
                self._log(f"Request error: {e}. Retrying...")
                time.sleep(self.base_delay + random.uniform(0, 0.5))

        sample["phenotype_inference"] = "error"
        return sample

    def run(self, samples: List[Dict]) -> List[Dict]:
        """Run phenotype inference on a list of samples using multithreading and safe retry."""
        self._log(f"Starting inference on {len(samples)} samples using {self.max_workers} workers.")

        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_sample = {executor.submit(self._infer_single_sample, sample): sample for sample in samples}

            for future in as_completed(future_to_sample):
                result = future.result()
                results.append(result)

        self._log("Inference completed.")
        return results