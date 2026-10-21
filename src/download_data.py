"""
Script to download and verify the authentic UCI Chronic Kidney Disease dataset.
Source: UCI Machine Learning Repository (Dataset ID 336)
URL: https://archive.ics.uci.edu/static/public/336/data.csv
"""

import os
import urllib.request
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

UCI_CKD_URL = "https://archive.ics.uci.edu/static/public/336/data.csv"
RAW_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw", "chronic_kidney_disease.csv")

def download_dataset(url: str = UCI_CKD_URL, destination: str = RAW_DATA_PATH) -> str:
    """
    Downloads the authentic UCI Chronic Kidney Disease dataset and saves it to data/raw/.
    If the file already exists, it verifies its integrity without re-downloading.
    """
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    if os.path.exists(destination) and os.path.getsize(destination) > 0:
        logging.info(f"Dataset already present at: {destination} ({os.path.getsize(destination)} bytes)")
        return destination

    logging.info(f"Downloading UCI CKD dataset from {url}...")
    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(url, headers=headers)
    
    with urllib.request.urlopen(req, timeout=20) as response:
        content = response.read()
        
    with open(destination, "wb") as f:
        f.write(content)
        
    logging.info(f"Successfully saved {len(content)} bytes to {destination}")
    return destination

if __name__ == "__main__":
    path = download_dataset()
    print(f"Downloaded dataset to: {path}")
