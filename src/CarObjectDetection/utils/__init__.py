import os
from  pathlib import Path
from  box import ConfigBox
import yaml
from CarObjectDetection import logger


def read_yaml(file_path:Path):
    try:
        with file_path.open('r') as file:
            config_file = yaml.safe_load(file)
            config=ConfigBox(config_file)
            logger.info(f" cofig files {config} loaded successfully")
        return config
    except Exception as e:
        logger.error(f"Failed to load config files {file_path}: {e}")
        raise e
    

def create_directories(file_paths:list,verbose=True):
    for file_path in file_paths:
        if not Path(file_path).exists():
            os.makedirs(file_path, exist_ok=True)
            if verbose:
                    logger.info(f"Created directory: {file_path}")
        else:
            if verbose:
                logger.info(f"All directories already exist")
