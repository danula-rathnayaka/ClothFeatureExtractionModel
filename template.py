import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

project_name = "ClothFeatureExtractor"


list_of_files = [
    ".github/workflows/.gitkeep",
    "data/raw/.gitkeep",
    "data/processed/.gitkeep",
    "data/augmented/.gitkeep",
    "data/splits/train/.gitkeep",
    "data/splits/val/.gitkeep",
    "data/splits/test/.gitkeep",
    "config/config.yaml",
    "dvc.yaml",
    "params.yaml",
    "requirements.txt",
    "setup.py",

    f"src/{project_name}/__init__.py",
    f"src/{project_name}/constants/__init__.py",
    f"src/{project_name}/entity/config_entity.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/scripts/data_ingestion.py",
    f"src/{project_name}/scripts/data_preprocessing.py",
    f"src/{project_name}/scripts/model_training.py",
    f"src/{project_name}/scripts/model_export.py",
    f"src/{project_name}/utils/util.py",

    "research/trials.ipynb",
    "models/saved_models/.gitkeep",
    "models/tflite/.gitkeep",
    "logs/.gitkeep"
]

if os.path.isfile('data'):
    os.remove('data')

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    # Create directories if they don't exist
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir}")

    # Create empty files if they don't exist
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} already exists")
