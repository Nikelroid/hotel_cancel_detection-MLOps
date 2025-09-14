import os
import pandas as pd
import joblib
from sklearn.model_selection import RandomizedSearchCV
import lightgbm as lgb 
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from config.model_params import *
from utils.common_functions import read_yaml,load_data
from scipy.stats import randint
import mlflow
import mlflow.sklearn

import warnings

warnings.simplefilter("ignore")
logger = get_logger(__name__)

class ModelTraining:
    def __init__(self,train_path,test_path,model_output_path):
        self.train_path = train_path
        self.test_path = test_path
        self.model_output_path = model_output_path

        self.params_dist = LIGHTGM_PARAMS
        self.random_search_params = RANDOM_SEARCH_PARAMS

    def load_and_split_data(self):
        try:
            logger.info(f'Loading train data from {self.train_path}')
            train_df = load_data(self.train_path)

            logger.info(f'Loading test data from {self.test_path}')
            test_df = load_data(self.test_path)

            logger.info('Spliting X and y columns for Train data')
            X_train = train_df.drop(columns='booking_status')
            y_train = train_df['booking_status']

            logger.info('Spliting X and y columns for Test data')
            X_test = test_df.drop(columns='booking_status')
            y_test = test_df['booking_status']

            logger.info('Data splitted successfully for train and test data')

            return X_train , X_test , y_train , y_test

        except Exception as e:
            logger.error(f"Error while loading data : {e}")
            raise CustomException ("Failed to load data",e)
        

    def train_lgbm(self,X_train,y_train):
        try:
            logger.info('Initializing the model')
            lgbm_model = lgb.LGBMClassifier(random_state=self.random_search_params['random_state'])

            logger.info('Starting the hyperparameter tuning step')
            random_search = RandomizedSearchCV(
                estimator=lgbm_model,
                param_distributions=self.params_dist,
                n_iter=self.random_search_params['n_iter'],
                cv=self.random_search_params['cv'],
                n_jobs=self.random_search_params['n_jobs'],
                verbose=self.random_search_params['verbose'],
                random_state=self.random_search_params['random_state'],
                scoring=self.random_search_params['scoring']
            )

            logger.info('Starting model training...')
            random_search.fit(X_train,y_train)
            logger.info('Hyperparameter tuning completed')

            logger.info('Selecting best model...')
            best_params = random_search.best_params_
            best_lgbm_model = random_search.best_estimator_
            logger.info(f'Best estimator and params selected: {best_params}')

            return best_lgbm_model
        
        except Exception as e:
            logger.error(f"Error while training model : {e}")
            raise CustomException ("Failed to train model ",e)
        

    def evaluate_model(self, model, X_test, y_test):
        try:
            logger.info("Evaluating the model...")
            y_pred = model.predict(X_test)

            logger.info("Scoring the model...")
            accuracy = accuracy_score(y_test,y_pred)
            precision = precision_score(y_test,y_pred)
            recall = recall_score(y_test,y_pred)
            f1 = f1_score(y_test,y_pred)

            logger.info(" ")
            logger.info(f"____________________________________")
            logger.info(f"Accuracy of model: {accuracy}")
            logger.info(f"Precision of model: {precision}")
            logger.info(f"Recall of model: {recall}")
            logger.info(f"F1 score of model: {f1}")
            logger.info(f"____________________________________")
            logger.info(" ")

            return {
                'accuracy' : accuracy,
                'precision' : precision,
                'recall' : recall,
                'f1': f1
            }
        
        except Exception as e:
            logger.error(f"Error while evaluating model : {e}")
            raise CustomException ("Failed to evaluate model ",e)
        
    
    def save_model(self,model):
        try:
            os.makedirs(os.path.dirname(self.model_output_path),exist_ok=True)

            logger.info("Saving the model")
            joblib.dump(model,self.model_output_path)
            logger.info(f"Model saved to {self.model_output_path}!")

        except Exception as e:
            logger.error(f"Error while saving model : {e}")
            raise CustomException ("Failed to save model ",e)
        
    def run(self):
        try:
            with mlflow.start_run():
                logger.info("Starting the model training pipeline...")
                logger.info("MLflow experimentation started")

                logger.info("Logging the training and testing dataset to MLflow")
                mlflow.log_artifact(self.train_path,artifact_path='datasets')
                mlflow.log_artifact(self.test_path,artifact_path='datasets')

                X_train , X_test , y_train , y_test = self.load_and_split_data()
                logger.info(X_train.columns)
                best_lgbm_model = self.train_lgbm(X_train,y_train)
                eval_metrics = self.evaluate_model(best_lgbm_model,X_test,y_test)
                self.save_model(best_lgbm_model)

                logger.info("Logging the model to MLflow")
                mlflow.log_artifact(self.model_output_path)

                logger.info("Logging params and metrics of the model to MLflow")
                mlflow.log_params(best_lgbm_model.get_params())
                mlflow.log_metrics(eval_metrics)


                logger.info("Model training pipeline is completed successfully")

        except Exception as e:
            logger.error(f"Error in train pipeline : {e}")
            raise CustomException ("Model training pipeline failed ",e)
        

if __name__ == '__main__':
    trainer = ModelTraining(
        train_path= PROCESSED_TRAIN_DATA_DIR,
        test_path= PROCESSED_TEST_DATA_DIR,
        model_output_path= MODEL_OUTPUT_PATH
    )
    trainer.run()




