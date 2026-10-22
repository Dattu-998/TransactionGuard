import argparse
from pathlib import Path
from transactionguard.utils.logging import get_logger
from transactionguard.config import config
logger = get_logger(__name__)

def download_kaggle_dataset(output_dir: Path):
    logger.info("Downloading dataset from Kaggle...")
    try:
        import kaggle
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files('mlg-ulb/creditcardfraud', path=str(output_dir), unzip=True)
        logger.info(f"Dataset downloaded successfully to {output_dir}")
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
if __name__ == "__main__":
    download_kaggle_dataset(config.DATA_DIR / "raw")
