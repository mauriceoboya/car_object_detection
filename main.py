from CarObjectDetection.constant.common import CONFIG_FILEPATH,PARAMS_FILEPATH
from CarObjectDetection.pipeline.model_training_pipeline import ModelTrainingPipeline


if __name__ == "__main__":
    config_manager = ModelTrainingPipeline()
    config_manager.main()
