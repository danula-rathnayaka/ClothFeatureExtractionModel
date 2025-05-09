from ClothFeatureExtractor.config.configuration import ConfigurationManager
from ClothFeatureExtractor import logger
from ClothFeatureExtractor.scripts.prepare_base_model import PrepareBaseModel

STAGE_NAME = "Prepare Base Model"


class PrepareBaseModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        prepare_base_model_config = config.get_prepare_base_model_config()
        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
        prepare_base_model.get_base_model()
        prepare_base_model.update_base_model()


if __name__ == '__main__':
    try:
        logger.info(f"========== Stage {STAGE_NAME} Started ==========")
        obj = PrepareBaseModelTrainingPipeline()
        obj.main()
        logger.info(f"========== Stage {STAGE_NAME} Completed ==========")
    except Exception as e:
        logger.error(f"========== Stage {STAGE_NAME} Exception Occurred ==========")
        logger.error(e)
        raise e
