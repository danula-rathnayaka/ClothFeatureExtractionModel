from ClothFeatureExtractor import path_to_root
from ClothFeatureExtractor.entity.config_entity import PrepareBaseModelConfig
import tensorflow as tf

from ClothFeatureExtractor.utils.util import save_model


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.model = None
        self.config = config

    def get_base_model(self):
        self.model = tf.keras.applications.EfficientNetB0(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top
        )

        save_model(path=path_to_root / self.config.base_model_path, model=self.model)

    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        # Freeze layers
        if freeze_all:
            for layer in model.layers:
                layer.trainable = False
        elif freeze_till is not None and freeze_till > 0:
            for layer in model.layers[:-freeze_till]:
                layer.trainable = False
            for layer in model.layers[-freeze_till:]:
                layer.trainable = True
        else:
            for layer in model.layers:
                layer.trainable = True

        # Shared trunk
        x = tf.keras.layers.GlobalAveragePooling2D()(model.output)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.4)(x)
        x = tf.keras.layers.Dense(256, activation='relu',
                                  kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)

        # Create each output head using its specific class count
        outputs = []
        for num_classes in classes:
            head = tf.keras.layers.Dense(128, activation='relu',
                                         kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
            head = tf.keras.layers.Dropout(0.3)(head)
            out = tf.keras.layers.Dense(num_classes, activation='softmax')(head)
            outputs.append(out)

        # Create the model
        full_model = tf.keras.models.Model(inputs=model.input, outputs=outputs)

        # Compile with separate losses for each output
        full_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=[tf.keras.losses.SparseCategoricalCrossentropy()] * len(classes),
            metrics=["accuracy"]
        )

        full_model.summary()
        return full_model

    def update_base_model(self, freeze_all=True, freeze_till=None, learning_rate=None):
        if learning_rate is None:
            learning_rate = self.config.params_learning_rate

        full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=freeze_all,
            freeze_till=freeze_till,
            learning_rate=learning_rate
        )

        save_model(path=path_to_root / self.config.updated_base_model_path, model=full_model)
        return full_model
