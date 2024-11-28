from CarObjectDetection import logger
from CarObjectDetection.utils import read_yaml, create_directories
from CarObjectDetection.constant.common import CONFIG_FILEPATH,PARAMS_FILEPATH
import numpy as np
import cv2
from PIL import Image
import os
import json
import pandas as pd
import yaml
import subprocess
from ultralytics import YOLO

class ConfigManager:
    def __init__(self, config_path, params_path):
        # Load configurations and parameters
        self.config = read_yaml(config_path)
        self.params = read_yaml(params_path)

        # Create required directories
        create_directories([
            self.config["model_training"]["root_dir"],
            self.config["model_training"]["model_coco_images"],
            self.config["model_training"]["model_coco_labels"],
        ])

    def add_border(self, image, border_size, color):
        """
        Adds a border to an image.
        """
        image_np = np.array(image)
        image_np = cv2.copyMakeBorder(
            image_np,
            border_size,
            border_size,
            border_size,
            border_size,
            cv2.BORDER_CONSTANT,
            value=color,
        )
        return Image.fromarray(image_np)

    def create_labels(self):
        """
        Converts JSON annotations to YOLO format and processes images with borders
        for test, train, and validation datasets.
        """
        # Dataset paths derived from the configuration
        dataset_folders = {
            "test": self.config["model_training"]["test_path"],
            "train": self.config["model_training"]["train_path"],
            "valid": self.config["model_training"]["valid_path"],
        }

        # Output paths for processed images and labels
        path_out_labels = self.config["model_training"]["model_coco_labels"]
        path_out_images = self.config["model_training"]["model_coco_images"]

        for dataset_type, path_in in dataset_folders.items():
            # Get all JSON files in the dataset directory
            json_files = [file for file in os.listdir(path_in) if file.endswith(".json")]

            if not json_files:
                raise FileNotFoundError(f"No JSON annotation files found in {path_in} for {dataset_type} dataset.")

            # Load the first JSON file
            with open(os.path.join(path_in, json_files[0]), "r") as f:
                data = json.load(f)

            # Extract and merge necessary data
            df_images = pd.DataFrame(data["images"]).rename(columns={"id": "image_id"})
            df_categories = pd.DataFrame(data["categories"]).rename(columns={"id": "category_id"})
            df_annotations = pd.DataFrame(data["annotations"])
            merged_df = (
                pd.merge(df_annotations, df_categories, on="category_id", how="left")
                .merge(df_images, on="image_id", how="left")
            )
            df = merged_df[["category_id", "bbox", "file_name"]]
            df[["bbox_x", "bbox_y", "bbox_width", "bbox_height"]] = pd.DataFrame(
                df["bbox"].tolist(), index=df.index
            )
            df.drop("bbox", axis=1, inplace=True)
            df.drop_duplicates(inplace=True)

            # Create directories for labels and images for the current dataset type
            dataset_labels_dir = os.path.join(path_out_labels, dataset_type)
            dataset_images_dir = os.path.join(path_out_images, dataset_type)
            os.makedirs(dataset_labels_dir, exist_ok=True)
            os.makedirs(dataset_images_dir, exist_ok=True)

            # Generate label files
            for file_name, group in df.groupby("file_name"):
                label_file = os.path.join(dataset_labels_dir, file_name.replace(".jpg", ".txt"))
                with open(label_file, "w") as f:
                    for _, row in group.iterrows():
                        class_index = row["category_id"]
                        x_center = row["bbox_x"] + row["bbox_width"] / 2
                        y_center = row["bbox_y"] + row["bbox_height"] / 2
                        width = row["bbox_width"]
                        height = row["bbox_height"]

                        # Normalize bounding box coordinates
                        x_center /= 320
                        y_center /= 320
                        width /= 320
                        height /= 320

                        f.write(f"{class_index} {x_center} {y_center} {width} {height}\n")

            print(f"Labels successfully created for {dataset_type} dataset.")

            # Process images and add borders
            for file_name in os.listdir(path_in):
                if file_name.endswith(".jpg"):
                    source_file = os.path.join(path_in, file_name)
                    destination_file = os.path.join(dataset_images_dir, file_name)
                    image = Image.open(source_file)
                    image_with_border = self.add_border(image, 20, (0, 0, 0))  # Default border size and color
                    image_with_border.save(destination_file)

            print(f"Images successfully processed for {dataset_type} dataset.")
    
    def train_model(self):
        folder_path= self.config.model_training.root_dir
        dataset_yaml_content = {
            "test": self.config["model_training"]["test_path"],
            "train": self.config["model_training"]["train_path"],
            "valid": self.config["model_training"]["valid_path"],
            "nc":36,
            'names': ['cars-bikes-people', 'Bus', 'Bushes', 'Person', 'Truck', 'backpack', 'bench',
             'bicycle', 'boat', 'branch', 'car', 'chair', 'clock', 'crosswalk', 'door',
             'elevator', 'fire_hydrant', 'green_light', 'gun', 'handbag', 'motorcycle',
             'person', 'pothole', 'rat', 'red_light', 'scooter', 'sheep', 'stairs', 'stop_sign',
             'suitcase', 'traffic light', 'traffic_cone', 'train', 'tree', 'truck', 'umbrella',
              'yellow_light']

        }
        dataset_yaml_path = os.path.join(folder_path, 'dataset.yaml')

        with open(dataset_yaml_path, 'w') as outfile:
            yaml.dump(dataset_yaml_content, outfile, default_flow_style=False)
            logger.info("Dataset.yaml file created successfully.")

        try:
            logger.info("Loading YOLO model...")
            yolo_model_path = self.config["model_training"]["model_path"]
            model = YOLO(yolo_model_path)  

            logger.info("Starting YOLO model training...")
            model.train(
                data=dataset_yaml_path,  
                imgsz=640,  
                epochs=2,  
                batch=16, 
                project=self.config["model_training"]["output_dir"], 
                name="yolo_training",  
                pretrained=True,  
            )
            logger.info("Model training completed successfully.")
        except Exception as e:
            logger.error(f"Error occurred during model training: {str(e)}")


if __name__ == "__main__":
    config_manager = ConfigManager(CONFIG_FILEPATH, PARAMS_FILEPATH)
    config_manager.train_model()
