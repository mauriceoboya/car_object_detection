from CarObjectDetection import logger
from CarObjectDetection.config.configurations import ConfigurationManager
import os
import yaml
from ultralytics import YOLO

class  ModelTrainingComponent(ConfigurationManager):
    def train_model(self):
        folder_path= self.config.model_training.root_dir
        train_path = self.config.model_training.train
        valid_path = self.config.model_training.val
        test_path = self.config.model_training.test
        dataset_yaml_content = {
            "train": os.path.abspath(train_path),
            "val": os.path.abspath(valid_path),
            "test": os.path.abspath(test_path),
            "nc": 37,
            "names": [
                'cars-bikes-people', 'Bus', 'Bushes', 'Person', 'Truck', 'backpack', 'bench',
                'bicycle', 'boat', 'branch', 'car', 'chair', 'clock', 'crosswalk', 'door',
                'elevator', 'fire_hydrant', 'green_light', 'gun', 'handbag', 'motorcycle',
                'person', 'pothole', 'rat', 'red_light', 'scooter', 'sheep', 'stairs', 'stop_sign',
                'suitcase', 'traffic light', 'traffic_cone', 'train', 'tree', 'truck', 'umbrella',
                'yellow_light'
            ]
        }

        dataset_yaml_path = os.path.join(folder_path, 'dataset.yaml')

        with open(dataset_yaml_path, 'w') as outfile:
            yaml.dump(dataset_yaml_content, outfile, default_flow_style=False)
            logger.info("Dataset.yaml file created successfully.")

        try:
            logger.info("Loading YOLO model...")
            yolo_model_path = self.config.model_training.model_path
            model = YOLO(yolo_model_path)  

            logger.info("Starting YOLO model training...")
            model.train(
                data=dataset_yaml_path,
                imgsz=self.params.imgsz,
                epochs=self.params.epochs,
                batch=self.params.batch_size,
                project=self.config.model_training.output_dir,
                name="yolo_training",
                pretrained=True,
                auto_augment="randaugment",
                #erasing=self.params.erasing,
                cfg= None,
                tracker="botsort.yaml"
            )
            logger.info("Model training completed successfully.")
        except Exception as e:
            logger.error(f"Error occurred during model training: {str(e)}")
