import pandas as pd
from pathlib import Path
from transactionguard.utils.logging import get_logger
logger = get_logger(__name__)
def load_data(filepath: Path) -> pd.DataFrame:
    if not filepath.exists():
        logger.error(f"Dataset file not found at: {filepath}")
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")
    logger.info(f"Loading data from {filepath}")
    return pd.read_csv(filepath)
