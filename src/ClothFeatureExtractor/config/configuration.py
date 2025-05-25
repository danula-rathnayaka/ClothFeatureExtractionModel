import os
from pathlib import Path

from ClothFeatureExtractor.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from ClothFeatureExtractor.utils.util import read_yaml, create_directories
from ClothFeatureExtractor.entity.config_entity import DataIngestionConfig, PrepareBaseModelConfig, TrainingConfig
from ClothFeatureExtractor import path_to_root


class ConfigurationManager:
    def __init__(
            self,
            config_filepath=path_to_root / CONFIG_FILE_PATH,
            params_filepath=path_to_root / PARAMS_FILE_PATH
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([path_to_root / self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([path_to_root / config.root_dir])

        return DataIngestionConfig(
            root_dir=config.root_dir,
            img_source_URL=config.img_source_URL,
            label_source_URL=config.label_source_URL,
            local_img_data_file=config.local_img_data_file,
            local_label_data_file=config.local_label_data_file,
            unzip_dir=config.unzip_dir
        )

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model

        create_directories([path_to_root / config.root_dir])

        return PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_classes=self.params.CLASSES
        )

    def get_training_config(self) -> TrainingConfig:
        training = self.config.training
        prepare_base_model = self.config.prepare_base_model
        params = self.params
        training_data = os.path.join(self.config.data_ingestion.unzip_dir, "data/images")
        create_directories([path_to_root / training.root_dir])

        label_paths = []

        for str_path in training.label_files:
            label_paths.append(Path(str_path))

        return TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            updated_base_model_path=Path(prepare_base_model.updated_base_model_path),
            training_data=Path(training_data),
            label_files=label_paths,
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_is_augmentation=params.AUGMENTATION,
            params_image_size=params.IMAGE_SIZE,
        )
