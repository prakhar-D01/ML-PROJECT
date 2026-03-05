import sys
import os
from dataclasses import dataclass

from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_model

@dataclass
class ModelTrainerConfig:
    training_model_file_path:str = os.path.join('artifacts', "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config= ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and testing input data")
            X_train, y_train, X_test, y_test= (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(), 
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'splitter':['best','random'],
                    'max_features':['sqrt','log2'],
                },
                "Random Forest Regressor":{
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    'max_features':['sqrt','log2',None],
                    'n_estimators':[50,100,200]
                },
                "Linear Regression":{},

                "K-Neighbors Regressor": {
                    'n_neighbors':[3,5,7,9],
                    'algorithm':['auto','ball_tree','kd_tree']
                },

                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators':[50,100,200]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    'loss':['linear','square','exponential'],
                    'n_estimators':[50,100,200]
                }
                
            }

            model_report:dict= evaluate_model(X_train=X_train, y_train=y_train,X_test= X_test, y_test=y_test, models=models,param=params)

            best_model_score= max(model_report.values())

            best_model_name= list(model_report.keys())[ list(model_report.values()).index(best_model_score) ]
            best_model= models[best_model_name]
            
            if best_model_score<0.6:
                raise CustomException("No Best model found", sys)
            
            logging.info("Best found model on both training and testing data")

            save_object(
                file_path= self.model_trainer_config.training_model_file_path,
                obj= best_model
            )

            predicted= best_model.predict(X_test)

            model_r2_score= r2_score(y_test, predicted)
            return model_r2_score

        except Exception as e:
            raise CustomException(e, sys)