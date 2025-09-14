from src.data_ingestion import DataIngestion
from src.data_preprocessing import DataProcessor
from src.model_training import ModelTraining
from utils.common_functions import read_yaml
from config.paths_config import *


if __name__ == '__main__' :

    #1.Data Ingestion
    data_ingestion = DataIngestion(read_yaml(CONFIG_PATH))
    data_ingestion.run()

    #1.Data Process
    processor = DataProcessor(TRAIN_FILE_PATH,TEST_FILE_PATH,PROCESSED_DIR,CONFIG_PATH)
    processor.precess()

    #1.Model Training
    trainer = ModelTraining(PROCESSED_TRAIN_DATA_DIR,PROCESSED_TEST_DATA_DIR,MODEL_OUTPUT_PATH)
    trainer.run()