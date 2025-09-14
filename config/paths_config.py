import os

####################  DATA_INGESTION   ######################


RAW_DIR = "artifacts/raw"
RAW_PATH_FILE = os.path.join(RAW_DIR, "raw.csv")
TRAIN_FILE_PATH = os.path.join(RAW_DIR, "train.csv")
TEST_FILE_PATH = os.path.join(RAW_DIR, "test.csv")

CONFIG_PATH = "config/config.yaml"


####################  DATA_PROCESSING   ######################

PROCESSED_DIR = "artifacts/processed"
PROCESSED_TRAIN_DATA_DIR = os.path.join(PROCESSED_DIR,"processed_train.csv")
PROCESSED_TEST_DATA_DIR = os.path.join(PROCESSED_DIR,"processed_test.csv")


####################  MODEL_TRAINING   ######################

MODEL_OUTPUT_PATH = "artifacts/models/lgbm_pickle.pkl"