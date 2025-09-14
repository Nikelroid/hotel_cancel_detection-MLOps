import os
import pandas as pd
import numpy as np
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common_functions import read_yaml,load_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder,PowerTransformer
from imblearn.over_sampling import SMOTE

logger = get_logger(__name__)

class DataProcessor:
    def __init__(self,train_path,test_path,processed_dir,config_path):
        self.train_path = train_path
        self.test_path = test_path
        self.processed_dir = processed_dir

        self.config = read_yaml(config_path)

        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def preprocess_data(self,df):
        try:
            logger.info("Start data processing step")

            logger.info("Dropping the columns")
            df.drop(columns=['Unnamed: 0', 'Booking_ID'],inplace=True)
            df.drop_duplicates(inplace=True)

            logger.info("Get categorical and numerical columns of dataset")
            cat_cols = self.config["data_processing"]["categorical_columns"]
            num_cols = self.config["data_processing"]["numerical_columns"]

            logger.info("Applying label encoding")
            label_encoder = LabelEncoder()
            mappings = {}
            for column in cat_cols:
                df[column] = label_encoder.fit_transform(df[column])
                mappings[column] = {label:code for label,code in zip(label_encoder.classes_,label_encoder.transform(label_encoder.classes_))}
            
            logger.info("Label mappings are : ")
            for col,mapping in mappings.items():
                logger.info(f"{col} -> {mapping}")

            logger.info("Doing skewness handling")
            skewness_fix_method = self.config["data_processing"]["skewness_fix_method"]
            # pt = PowerTransformer(method=skewness_fix_method, standardize=True)
            # df[num_cols] = pt.fit_transform(df[num_cols])
            return df
        
        except Exception as e:
            logger.error(f"Error during process {e}")
            raise CustomException("Error in preprocessing deta",e)
        
    def balance_data(self,df):
        try:
            logger.info('Handlig imbalance data...')
            X = df.drop(columns='booking_status')
            y = df['booking_status']

            smote =SMOTE(random_state=42)
            X_res , y_res =smote.fit_resample(X,y)

            balanced_df = pd.DataFrame(X_res,columns=X.columns)
            balanced_df['booking_status'] = y_res

            logger.info("Data balanced successfully")
            return balanced_df

        except Exception as e:
            logger.error(f"Error during process {e}")
            raise CustomException("Error in balancing deta",e)
        
     
    def select_features(self,df):
        try:
            logger.info('Starting feature selection...')

            X = df.drop(columns='booking_status')
            y = df['booking_status']

            model = RandomForestClassifier(random_state= 42)
            model.fit(X,y)

            feature_importance = model.feature_importances_
            feature_importance_df = pd.DataFrame({
                                    'feature':X.columns,
                                    'importance':feature_importance
                                     })
            
            no_of_features_top = self.config["data_processing"]["no_of_features"]
            top_feature_importance_df = feature_importance_df.sort_values(by='importance',ascending=False)
            top_10_features = top_feature_importance_df["feature"].head(no_of_features_top).values
            top_10_df = df[top_10_features.tolist()+['booking_status']]
            logger.info(f"Feature selection completed with {no_of_features_top} features")
            logger.info(f"Features selected:{top_10_features}")
            return top_10_df

        except Exception as e:
            logger.error(f"Error during feature selection step, {e}")
            raise CustomException("Error while feature selection",e)
        

    def save_data(self,df,file_path):
        try:
            logger.info(f"Saving our data inside {file_path}")
            df.to_csv(file_path,index=False)
            logger.info(f"Data saved successfully to {file_path}")

        
        except Exception as e:
            logger.error(f"Error during saving data, {e}")
            raise CustomException("Error while saving data",e)
        
    def precess(self):
        try:
            logger.info("Loading data from raw directory")

            train_df = load_data(self.train_path)
            test_df = load_data(self.test_path)

            train_df = self.preprocess_data(train_df)
            test_df = self.preprocess_data(test_df)

            train_df = self.balance_data(train_df)
            test_df = self.balance_data(test_df)

            train_df = self.select_features(train_df)
            test_df = test_df[train_df.columns]

            self.save_data(train_df,PROCESSED_TRAIN_DATA_DIR)
            self.save_data(test_df,PROCESSED_TEST_DATA_DIR)
        except Exception as e:
            logger.error(f"Error in preprocessing pipeline, {e}")
            raise CustomException("Error in preprocessing pipeline",e)


if __name__ == "__main__":
    processor = DataProcessor(train_path=TRAIN_FILE_PATH,
                              test_path=TEST_FILE_PATH,
                              processed_dir=PROCESSED_DIR,
                              config_path=CONFIG_PATH)
    processor.precess()



        


