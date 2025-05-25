import os
from collections import defaultdict
from pathlib import Path
import tensorflow as tf

from ClothFeatureExtractor import path_to_root
from ClothFeatureExtractor.entity.config_entity import TrainingConfig
from ClothFeatureExtractor.utils.util import save_model


class Training:
    def __init__(self, config: TrainingConfig):
        self.validation_steps = None
        self.steps_per_epoch = None
        self.valid_generator = None
        self.train_generator = None
        self.config = config

    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )

    def _parse_label_file(self, label_file_path: Path) -> dict:
        label_map = {}
        with open(path_to_root / label_file_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                filename = parts[0]
                labels = list(map(int, parts[1:]))
                label_map[filename] = labels
        return label_map

    def _load_dataset(self, image_dir, label_map, split="train"):
        image_paths = list((path_to_root / Path(image_dir)).glob("*.jpg"))

        # Optional: shuffle and split
        total = len(image_paths)
        split_index = int(0.8 * total)

        if split == "train":
            image_paths = image_paths[:split_index]
        else:
            image_paths = image_paths[split_index:]

        def load_and_preprocess(path):
            image = tf.io.read_file(path)
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.image.resize(image, self.config.params_image_size[:-1])
            image = image / 255.0

            if self.config.params_is_augmentation:
                image = tf.image.random_flip_left_right(image)
                image = tf.image.random_brightness(image, max_delta=0.1)
                image = tf.image.random_contrast(image, 0.9, 1.1)

            filename = tf.strings.split(path, os.sep)[-1]
            label = tf.py_function(lambda f: label_map[f.numpy().decode()], [filename], tf.int32)
            label.set_shape([len(next(iter(label_map.values())))])

            # Split label vector into list of scalars, one per output
            label_list = tf.unstack(label)

            return image, label_list

        ds = tf.data.Dataset.from_tensor_slices(image_paths)
        ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(self.config.params_batch_size).prefetch(tf.data.AUTOTUNE)

        return ds

    def train_valid_generator(self):
        label_map = defaultdict(list)

        for label_file in self.config.label_files:
            file_data = self._parse_label_file(label_file)
            for image_name, labels in file_data.items():
                label_map[image_name].extend(labels)

        self.train_generator = self._load_dataset(self.config.training_data, label_map, split="train")
        self.valid_generator = self._load_dataset(self.config.training_data, label_map, split="val")

    def train(self):

        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator
        )

        save_model(
            path=path_to_root / self.config.trained_model_path,
            model=self.model
        )
