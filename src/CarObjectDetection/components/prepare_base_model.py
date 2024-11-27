import os
from pathlib import Path
from ultralytics import YOLO
from CarObjectDetection.config.configurations import ConfigurationManager
from CarObjectDetection import logger


class BaseModelPreparation(ConfigurationManager):
    def download_pretrained_model(self):
        """Downloads and saves the pre-trained YOLO model."""
        model_name = self.model_config.model_name
        model_dir = Path(self.model_config.model_dir)

        try:
            logger.info(f"Downloading YOLO model: {model_name}")
            model = YOLO(model_name) 
            model.save(model_dir / f"{model_name}.pt")  
            logger.info(f"Model {model_name} downloaded and saved to {model_dir}")
        except Exception as e:
            logger.error(f"Failed to download the model {model_name}: {e}")
            raise e

    def get_model_path(self) -> Path:
        """Returns the path to the saved pre-trained model."""
        model_name = self.model_config.model_name
        return Path(self.model_config.model_dir) / f"{model_name}.pt"


