from CarObjectDetection import logger
from CarObjectDetection.components.data_ingestion_component import DataIngestionComponent


STAGE_ONE="Data Ingestion Pipeline"

class DataIngestionPipeline:
    def __init__(self):
        pass
    def main(self):
        logger.info(f"Starting {STAGE_ONE}...")
        data_ingestion_component = DataIngestionComponent()
        data_ingestion_component.download_data()
        logger.info(f"{STAGE_ONE} completed successfully.")

if __name__ == "__main__":
    data_ingestion_pipeline = DataIngestionPipeline()
    data_ingestion_pipeline.main()
