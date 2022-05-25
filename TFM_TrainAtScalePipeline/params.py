### GCP Storage - - - - - - - - - - - - - - - - - - - - - -

BUCKET_NAME = 'wagon-data-871-bidermane'

##### Data  - - - - - - - - - - - - - - - - - - - - - - - -

# train data file location
BUCKET_TRAIN_DATA_PATH = 'data/train_1k.csv'
# test data file location
BUCKET_TEST_DATA_PATH = 'data/test.csv'

##### Model - - - - - - - - - - - - - - - - - - - - - - - -

# model folder name (will contain the folders for all trained model versions)
MODEL_NAME = 'taxifare'

# model version folder name (where the trained model.joblib file will be stored)
MODEL_VERSION = 'v2'

STORAGE_LOCATION = 'models/simpletaxifare/model.joblib'

#------------------MLFlow ------------------
MLFLOW_URI = "https://mlflow.lewagon.ai/"
EXPERIMENT_NAME = "first_experiment"
