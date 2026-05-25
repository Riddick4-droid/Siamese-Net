import os
import shutil
from pathlib import Path
from typing import Dict, Any

import kagglehub

from src.logger import get_logger
from src.exceptions import DataIngestionException

logger = get_logger(__name__)

def download_and_cache_dataset(config: Dict[str,Any])->Path:
    """
    Download the LFW dataset from Kagglehub (if not already cached) and
    make it available under a stable local directory.

    This function creates a symlinked or copied cache so that subsequent
    runs skip the download entirely.

    Args:
        config: full configuration dictionary loaded from YAML.

    Returns:
        path to the root directory containing person subfolders.

    Raises:
        ProjectException: If the expected image folder cannot be located inside
            the downloaded archive.
    """
    dataset_name = config["data"]["dataset_name"]
    cache_dir = Path(config["data"]["cache_dir"]).expanduser().resolve()
    data_dir = cache_dir / "lfw/lfw-deepfunneled"

    #if we already have the images cached, just return it
    if data_dir.exists() and any(data_dir.iterdir()):
        logger.info("Dataset already cached at %s", data_dir)
        return data_dir
    logger.info("Downloading dataset '%s' from kagglehub...",dataset_name)
    kaggle_path = kagglehub.dataset_download(dataset_name)
    logger.info("kagglehub cache path: %s", kaggle_path)

    #locate the directory that contains the person subfolder
    #common names are "lfw-deepfunneled" or "lfw". walk the tree
    src_dir = None
    for root, dirs, _, in os.walk(kaggle_path):
        if "lfw-deepfunneled" in dirs:
            src_dir = Path(root) / "lfw-deepfunneled"
            break
        if "lfw" in dirs:
            candidate = Path(root) / "lfw"
            #make sure its the image folder (contains subdirectories with images)
            if any((candidate/d).is_dir() for d in os.listdir(candidate)):
                src_dir = candidate
                break
    if src_dir is None:
        raise DataIngestionException(
            "Could not locate the image directory inside the downloaded dataset. "
            f"Please inspect the Kagglehub cache at {kaggle_path} manually."
        )
    logger.info("found source images at %s", src_dir)

    #populate the stable cache with symlinks (fall baclk to copy)
    data_dir.mkdir(parents=True, exist_ok=True)
    for item in src_dir.iterdir():
        src_item = src_dir / item.name
        dst_item = data_dir / item.name
        if dst_item.exists():
            continue
        try:
            os.symlink(src_item, dst_item)
        except OSError:
            logger.debug("symlink failed for %s,falling back to copy.", item.name)
            if src_item.is_dir():
                shutil.copytree(src_item, dst_item)
            else:
                shutil.copy2(src_item,dst_item)
    logger.info("dataset ready at %s", data_dir)
    return data_dir

if __name__ == "__main__":
    from src.utils import load_config
    config = load_config(config_path="./config.yaml")
    

    #run the function
    download_and_cache_dataset(config)