from CarObjectDetection import logger
from CarObjectDetection.utils import read_yaml, create_directories
from CarObjectDetection.constant.common import CONFIG_FILEPATH, PARAMS_FILEPATH
from CarObjectDetection.entity.config_entity import DataIngestionConfig
s


class ConfigurationManager:
    def __init__(self, config=CONFIG_FILEPATH, params=PARAMS_FILEPATH):
        self.config = read_yaml(config)
        self.params = read_yaml(params)
        try:
            artifacts_root = self.config.artifacts_root
            create_directories([artifacts_root])
        except AttributeError:
            logger.error(f"Key 'artifacts_root' not found in configuration file: {config}")
            raise

    def get_data_ingestion(self) -> DataIngestionConfig:
        try:
            ingestion_config = DataIngestionConfig(
                root_dir=self.config.data_ingestion.root_dir,
                source_URL=self.config.data_ingestion.source_URL,
                local_data_dir=self.config.data_ingestion.local_data_dir,
                unzip_dir=self.config.data_ingestion.unzip_dir
            )
            create_directories([ingestion_config.root_dir])
            return ingestion_config
        except AttributeError:
            logger.error(f"Key 'data_ingestion.root_dir' not found in configuration file.")
            raise

