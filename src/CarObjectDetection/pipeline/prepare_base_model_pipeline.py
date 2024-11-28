from CarObjectDetection import logger
from CarObjectDetection.components.prepare_base_model import BaseModelPreparation

STAGE2="base model"
class BaseModelPreparationPipeline:
    def __init__(self):
        pass
    def main(self):
        logger.info(f"Starting {STAGE2}...")
        base_model_preparation = BaseModelPreparation()
        base_model_preparation.prepare_base_model()
        logger.info(f"{STAGE2} completed successfully.")



if __name__ == "__main__":
    data_base_model = BaseModelPreparationPipeline()
    data_base_model.main()