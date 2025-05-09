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
            os.makedirs(path_to_root / "data/raw", exist_ok=True)
            url_prefix = 'https://drive.google.com/uc?/export=download&id='

            for resource in [[self.config.img_source_URL, self.config.local_img_data_file],
                             [self.config.label_source_URL, self.config.local_label_data_file]]:
                logger.info(
                    f"Downloading data from {resource[0]} into file {resource[1]}")

                img_file_id = resource[0].split("/")[-2]
                gdown.download(url_prefix + img_file_id, str(path_to_root / resource[1]))

                logger.info(
                    f"Downloaded data from {resource[0]} into file {resource[1]}")

        except Exception as e:
            logger.error("Error while downloading the drive data")
            raise e

    def extract_zip_file(self):
        unzip_path = path_to_root / self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(path_to_root / self.config.local_img_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
