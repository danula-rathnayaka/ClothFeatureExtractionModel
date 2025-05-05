import os
import zipfile

import gdown
from ClothFeatureExtractor import logger
from ClothFeatureExtractor.entity.config_entity import DataIngestionConfig
from ClothFeatureExtractor import path_to_root


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self) -> None:
        try:
            os.makedirs(path_to_root / "artifacts/data_ingestion", exist_ok=True)
            logger.info(f"Downloading data from {self.config.source_URL} into file {self.config.local_data_file}")

            file_id = self.config.source_URL.split("/")[-2]
            prefix = 'https://drive.google.com/uc?/export=download&id='
            gdown.download(prefix + file_id, str(path_to_root / self.config.local_data_file))

            logger.info(f"Downloaded data from {self.config.source_URL} into file {self.config.local_data_file}")

        except Exception as e:
            logger.error("Error while downloading the drive data")
            raise e

    def extract_zip_file(self):
        unzip_path = path_to_root / self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(path_to_root / self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
