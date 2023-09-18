MLFLOW_URI = "https://mlflow.lewagon.ai/"
EXPERIMENT_NAME = "[IND] [BALI] [cherifbenham] model name + version"

# imports

from folder.dataa import get_data
from folder.dataa import clean_data

from folder.encoders import TimeFeaturesEncoder, DistanceTransformer

from folder.utils import compute_rmse

import pandas as pd

import mlflow


from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.compose import ColumnTransformer


class Trainer():
    def __init__(self, X, y):
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        pipe_time = make_pipeline(TimeFeaturesEncoder(time_column='pickup_datetime'), StandardScaler())
        pipe_distance = make_pipeline(DistanceTransformer(), RobustScaler())

        # column transformer
        dist_cols = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']
        time_cols = ['pickup_datetime']

        bloc = ColumnTransformer([('time', pipe_time, time_cols), ('distance', pipe_distance, dist_cols)])

        # workflow
        self.pipeline = Pipeline(steps=[('bloc', bloc), ('regressor', LinearRegression())])
            
    def run(self):
        self.pipeline.fit(self.X, self.y)

    def evaluate(self, X_test, y_test):
        pipe = self.pipeline
        y_pred = pipe.predict(X_test)
        self.mlflow_log_param(param_name, param_value)
        self.mlflow_log_metric(metric_name, metric_value)
        return compute_rmse(y_pred, y_test)
    
    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)
    
if __name__ == "__main__":
    
    # get data
    df = get_data(n_rows=10000)
    
    # clean data
    df = clean_data(df)
    
    # set X and y
    X = df.drop(columns='fare_amount')
    y = df['fare_amount'] 

    # hold out
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)    
    
    # train
    trainer = Trainer(X_train, y_train)
    trainer.set_pipeline()
    trainer.run()
    
    # evaluate
    print(trainer.evaluate(X_test,y_test))