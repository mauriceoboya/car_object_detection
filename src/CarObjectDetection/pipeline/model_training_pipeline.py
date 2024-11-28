from CarObjectDetection import logger
from CarObjectDetection.components.model_training_component import ModelTrainingComponent

STAGE3="Model training"
class ModelTrainingPipeline:
    def __init__(self):
        pass
    def main(self):
        logger.info(f"Starting {STAGE3}...")
        model_training_pipeline = ModelTrainingComponent()
        model_training_pipeline.train_model()
        logger.info(f"{STAGE3} completed successfully.")
