import os
import argparse
import pandas as pd
import joblib
import yaml
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from azureml.core import Workspace, Dataset, Datastore, Run
from azureml.core.authentication import InteractiveLoginAuthentication
import logging
from azureml.core import Run
# Configure logging
logging.basicConfig(level=logging.INFO)

def load_config(config_path):
    """Load configuration settings from a YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def initialize_workspace(config):
    """Initialize Azure ML workspace."""
    auth = InteractiveLoginAuthentication(tenant_id=config['azure']['tenant_id'], force=True)
    ws = Workspace(subscription_id=config['azure']['subscription_id'],
                   resource_group=config['azure']['resource_group'],
                   workspace_name=config['azure']['workspace_name'], auth=auth)
    logging.info(f"Workspace {ws.name} initialized successfully.")
    return ws, Datastore.get(ws, config['azure']['datastore_name'])

def load_data(datastore, file_path):
    """Load data from Azure ML datastore."""
    dataset = Dataset.Tabular.from_delimited_files(path=[(datastore, file_path)])
    return dataset.to_pandas_dataframe()

def upload_to_datastore(datastore, src_dir, target_path=""):
    """Uploads a file from a specified directory to the Azure ML Datastore.

    Args:
        datastore (Datastore): The Azure ML Datastore object.
        src_dir (str): The source directory containing files to upload.
        target_path (str): The target path in the datastore where files will be uploaded.
    """
    datastore.upload(src_dir=src_dir, target_path=target_path, overwrite=True)
    logging.info("Data uploaded to datastore")
    
def train_model(run, df, config):
    """Train and evaluate the model, and save the outputs."""
    y = df[config['model']['target']]
    X = df.drop(config['model']['target'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config['model']['test_size'], random_state=config['model']['random_state'])

    #Exporting the file
    path = "tmp/"
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of directory %s failed" % path)
    else:
        print("Sucessfully created the directory %s " % path)
    
    temp_path = path + "training.csv"
    df.to_csv(temp_path)
        
    for n in config['model']['hyperparameters']['n_estimators']:
        model = RandomForestClassifier(n_estimators=n)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = math.sqrt(mean_squared_error(y_test, y_pred))
        logging.info(f"Model with {n} estimators: RMSE = {rmse}")
        
        # Additional logging can be placed here for metrics

        filename = f"tmp/model_estimator_{n}.pkl"
        joblib.dump(value=model, filename=filename)
        # Assume run context is available and uploading to run context here
        run.upload_file(name=f"model_estimator_{n}.pkl", path_or_stream=filename)

def main(input_file):
    run = Run.get_context()

    config = load_config('config.yaml')
    ws, datastore = initialize_workspace(config)
    df = load_data(datastore, input_file)
    train_model(run, df, config)
    upload_to_datastore(datastore, "tmp/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model using Azure ML and hyperparameters from a YAML file.")
    parser.add_argument("--input-file", type=str, required=True, help="Path to the input data file")
    args = parser.parse_args()
    main(args.input_file)
