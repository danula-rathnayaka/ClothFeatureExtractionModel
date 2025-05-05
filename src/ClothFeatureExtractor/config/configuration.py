from ClothFeatureExtractor.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from ClothFeatureExtractor.utils.util import read_yaml, create_directories
from ClothFeatureExtractor.entity.config_entity import DataIngestionConfig
from ClothFeatureExtractor import path_to_root


class ConfigurationManager:
    def __init__(
            self,
            config_filepath=path_to_root / CONFIG_FILE_PATH,
            params_filepath=path_to_root / PARAMS_FILE_PATH
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        return DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir
        )
