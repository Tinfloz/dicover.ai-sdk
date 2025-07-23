import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from dotenv import load_dotenv
import os

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class PhenotypeInferencer:
    def __init__(self, max_workers=5):
        self.api_url = os.getenv("API_URL")
        self.api_key = os.getenv("API_KEY_GPT")
        self.deployment_id = os.getenv("DEPLOYMENT_ID")
        self.max_workers = max_workers

    def inspect_sample_metadata(self, sample):
        """Helper function to inspect what metadata is available in a sample"""
        self._log(f"Inspecting sample {sample.name}:")
        for key, value in sample.metadata.items():
            if isinstance(value, list):
                value_str = " | ".join(str(v) for v in value[:3]) 
                if len(value) > 3:
                    value_str += f" ... ({len(value)} total items)"
            else:
                value_str = str(value)
            self._log(f"  {key}: {value_str}")
    
    def _log(self, message):
        logging.info(message)

    def _convert_sample_to_dict(self, sample):
        try:
            # Extract all relevant metadata
            sample_info = {
                "id": sample.name,
                "title": sample.metadata.get("title", [""])[0] if sample.metadata.get("title") else "",
                "description": sample.metadata.get("description", [""])[0] if sample.metadata.get("description") else "",
            }
            
            # Add all other metadata keys that are informative
            important_keys = [
                "source_name_ch1", "organism_ch1", "characteristics_ch1", 
                "treatment_protocol_ch1", "growth_protocol_ch1", 
                "extract_protocol_ch1", "label_protocol_ch1",
                "hyb_protocol", "scan_protocol", "data_processing",
                "platform_id", "contact_name", "contact_email",
                "contact_institute", "contact_address", "contact_city",
                "contact_state", "contact_zip/postal_code", "contact_country",
                "supplementary_file", "series_id", "status", "submission_date",
                "last_update_date", "type", "channel_count", "taxid_ch1"
            ]
            
            for key in important_keys:
                if key in sample.metadata:
                    value = sample.metadata[key]
                    if isinstance(value, list):
                        sample_info[key] = " | ".join(str(v) for v in value if v)
                    else:
                        sample_info[key] = str(value)
            
            for key, value in sample.metadata.items():
                if key not in important_keys and key not in ["title", "description"]:
                    if isinstance(value, list):
                        sample_info[key] = " | ".join(str(v) for v in value if v)
                    else:
                        sample_info[key] = str(value)
            
            return sample_info
        except Exception as e:
            raise ValueError(f"Error converting sample {sample.name}: {e}")

    def _infer_single_sample(self, sample):
        try:
            sample_dict = self._convert_sample_to_dict(sample)
            
            prompt_parts = [
                "Based on the following sample metadata, classify whether this is a CONTROL or DISEASED sample.",
                "Look for keywords that indicate disease status, treatment conditions, or experimental groups.",
                "",
                f"Sample ID: {sample_dict['id']}"
            ]
            
            if sample_dict.get('title'):
                prompt_parts.append(f"Title: {sample_dict['title']}")
            
            if sample_dict.get('description'):
                prompt_parts.append(f"Description: {sample_dict['description']}")
            
            # Add other important metadata
            for key, value in sample_dict.items():
                if key not in ['id', 'title', 'description'] and value:
                    clean_key = key.replace('_ch1', '').replace('_', ' ').title()
                    prompt_parts.append(f"{clean_key}: {value}")
            
            prompt_parts.extend([
                "",
                "Common indicators:",
                "- CONTROL: normal, healthy, control, untreated, wild-type, baseline",
                "- DISEASED: disease, cancer, tumor, treated, mutant, patient, pathological",
                "",
                "Return either 'CONTROL' or 'DISEASED' based on the metadata above."
            ])
            
            prompt = "\n".join(prompt_parts)
            
            headers = {
                "Content-Type": "application/json",
                "api-key": self.api_key
            }

            payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0,
                "max_tokens": 50  
            }

            if self.deployment_id:
                url = f"{self.api_url}/deployments/{self.deployment_id}/chat/completions?api-version=2024-02-01"
            else:
                url = self.api_url

            response = requests.post(url, headers=headers, json=payload, timeout=30)

            if response.status_code == 200:
                reply = response.json()["choices"][0]["message"]["content"].strip()
                
                reply_upper = reply.upper()
                if "CONTROL" in reply_upper:
                    reply = "CONTROL"
                elif "DISEASED" in reply_upper:
                    reply = "DISEASED"
                else:
                    self._log(f"Unexpected response for {sample.name}: {reply}")
                    reply = "UNKNOWN"
                
                return sample.name, reply

            elif response.status_code == 429:
                self._log(f"Rate limit hit for {sample.name}. Retrying after delay.")
                time.sleep(5)  # Increased delay for rate limiting
                return self._infer_single_sample(sample)

            else:
                self._log(f"Error for {sample.name}: {response.status_code} - {response.text}")
                return sample.name, None

        except requests.exceptions.Timeout:
            self._log(f"Timeout for {sample.name}. Retrying...")
            time.sleep(1)
            return self._infer_single_sample(sample)
        except Exception as e:
            self._log(f"Unhandled exception for {getattr(sample, 'name', 'unknown')}: {e}")
            return getattr(sample, 'name', 'unknown'), None

    def run(self, samples: dict):
        results = []

        self._log(f"Starting phenotype inference for {len(samples)} samples...")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_sample = {
                executor.submit(self._infer_single_sample, sample): sample_id
                for sample_id, sample in samples.items()
            }

            completed_count = 0
            for future in as_completed(future_to_sample):
                sample_id = future_to_sample[future]
                try:
                    sample_name, result = future.result()
                    results.append((sample_name, result))
                    completed_count += 1
                    if completed_count % 10 == 0: 
                        self._log(f"Completed {completed_count}/{len(samples)} samples")
                except Exception as e:
                    self._log(f"Exception on {sample_id}: {e}")
                    results.append((sample_id, None))
                    completed_count += 1

        self._log("Inference completed.")
        return results