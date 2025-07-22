import os
import torch
import logging
from transformers import pipeline
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class PhenotypeInferencer:
    def __init__(self, use_gpu=True, max_workers=None, labels=None, verbose=True):
        self.device = 0 if use_gpu and torch.cuda.is_available() else -1
        self.max_workers = max_workers or os.cpu_count()
        self.labels = labels or ["control", "diseased", "unknown"]
        self.verbose = verbose

        if self.verbose:
            logging.info(f"Using device: {'cuda' if self.device == 0 else 'cpu'}")
            logging.info(f"Max workers: {self.max_workers}")

        model_name = (
            "facebook/bart-large-mnli" if self.device == 0
            else "MoritzLaurer/deberta-v3-base-zeroshot-v1"
        )

        if self.verbose:
            logging.info(f"Using model: {model_name}")

        self.classifier = pipeline(
            "zero-shot-classification",
            model=model_name,
            device=self.device
        )

    def _construct_metadata_text(self, gsm):
        metadata_text = []
        for key, value in gsm.metadata.items():
            if isinstance(value, list):
                metadata_text.append(" ".join(value))
            elif isinstance(value, str):
                metadata_text.append(value)
        final_text = " ".join(metadata_text).strip()
        if self.verbose:
            logging.info(f"Metadata for {gsm.name[:10]}...: {final_text[:80]}...")
        return final_text

    def _classify_sample(self, gsm_name, gsm):
        metadata_text = self._construct_metadata_text(gsm)

        if not metadata_text:
            if self.verbose:
                logging.warning(f"No metadata for {gsm_name}")
            return gsm_name, {"label": "unknown", "score": 0.0}

        try:
            result = self.classifier(metadata_text, self.labels)
            label = result["labels"][0]
            score = result["scores"][0]
            if self.verbose:
                logging.info(f"Classified {gsm_name}: {label} ({score:.2f})")
            return gsm_name, {"label": label, "score": score}
        except Exception as e:
            logging.error(f"Error classifying {gsm_name}: {e}")
            return gsm_name, {"label": "unknown", "score": 0.0}

    def infer(self, gse):
        phenotype_map = {}
        gsm_entries = list(gse.gsms.items())

        if self.verbose:
            logging.info(f"Starting classification for {len(gsm_entries)} samples...")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._classify_sample, gsm_name, gsm): gsm_name
                for gsm_name, gsm in gsm_entries
            }

            for future in as_completed(futures):
                gsm_id, result = future.result()
                phenotype_map[gsm_id] = result

        if self.verbose:
            label_counts = {label: sum(1 for x in phenotype_map.values() if x["label"] == label) for label in self.labels}
            logging.info("Classification complete. Summary:")
            for label, count in label_counts.items():
                logging.info(f" - {label}: {count}")

        return phenotype_map
