from CarObjectDetection import logger
from CarObjectDetection.config.configurations import ConfigurationManager
from kaggle.api.kaggle_api_extended import KaggleApi




class DataIngestionComponent:
    def __init__(self):
        self.config_manager = ConfigurationManager()

    def download_data(self):
        data_ingestion_config = self.config_manager.get_data_ingestion()

        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(data_ingestion_config.source_URL, path=data_ingestion_config.local_data_dir, unzip=True)
        logger.info(f"Data downloaded successfully from {data_ingestion_config.source_URL}")
