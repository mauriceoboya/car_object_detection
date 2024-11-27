from pathlib import Path
import os
import logging

project_name="CarObjectDetection"

file_paths=[
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/constant/common.py",
    f"src/{project_name}/config/configurations.py",
    f"src/{project_name}/entity/config_entity.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/pipeline/__init__.py",
    "main.py",
    "setup.py",
    "config/config.yaml",
    "params.yaml",
    ".github/workflows/.gitkeep"
]

for file_path in file_paths:
    file_path=Path(file_path)
    file_dir,file_name=os.path.split(file_path)
    if not Path(file_dir).exists():
        os.makedirs(file_dir, exist_ok=True)
        logging.info(f"Created directory: {file_dir}")
    else:
        logging.info(f"Directory already exists: {file_dir}")

    if not file_path.exists():
        with open(file_path, 'w') as f:
            pass
        logging.info(f"Created file: {file_path}")
    else:
        logging.info(f"File already exists: {file_path}")

