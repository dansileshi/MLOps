import os
import argparse
import pandas as pd
import yaml
from sklearn.preprocessing import QuantileTransformer
from azureml.core import Workspace, Datastore, Dataset, Run
from azureml.core.authentication import InteractiveLoginAuthentication
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    
def get_datastore(workspace, datastore_name):
    """Fetches the Datastore from the Azure ML Workspace."""
    datastore = Datastore.get(workspace, datastore_name)
    logging.info(f"Retrieved datastore: {datastore.name}")
    return datastore

def load_config(file_path='config.yaml'):
    """Loads configuration settings from a YAML file.
    
    Args:
        file_path (str): Path to the YAML configuration file.

    Returns:
        dict: Configuration dictionary.

    Raises:
        FileNotFoundError: If the YAML file is not found.
        yaml.YAMLError: If there is an error parsing the YAML file.
    """
    try:
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        logging.error("Configuration file not found.")
        raise
    except yaml.YAMLError as exc:
        logging.error("Error parsing YAML file: %s", exc)
        raise

def initialize_workspace(config):
    """Initializes and retrieves the Azure ML Workspace object using the configuration.

    Args:
        config (dict): Configuration dictionary containing Azure connection parameters.

    Returns:
        Workspace: An instantiated Azure ML Workspace object.
    """
    interactive_auth = InteractiveLoginAuthentication(tenant_id=config['azure']['tenant_id'], force=True)
    ws = Workspace(
        subscription_id=config['azure']['subscription_id'],
        resource_group=config['azure']['resource_group'],
        workspace_name=config['azure']['workspace_name'],
        auth=interactive_auth
    )
    logging.info(f"Workspace {ws.name} in resource group {ws.resource_group} located at {ws.location}")
    return ws


def load_data(datastore, file_path):
    """Loads data from Azure ML datastore and returns it as a pandas DataFrame."""
   

    df = Dataset.Tabular.from_delimited_files(path=[(datastore, file_path)]).to_pandas_dataframe()
    logging.info(f"Data loaded with shape: {df.shape}")
    return df

def preprocess_data(df):
    """Processes the DataFrame by cleaning and transforming the data."""
    df = df.drop_duplicates()
    df['Glucose'] = df['Glucose'].replace(0, df['Glucose'].mean())
    df['BloodPressure'] = df['BloodPressure'].replace(0, df['BloodPressure'].mean())
    df['SkinThickness'] = df['SkinThickness'].replace(0, df['SkinThickness'].median())
    df['Insulin'] = df['Insulin'].replace(0, df['Insulin'].median())
    df['BMI'] = df['BMI'].replace(0, df['BMI'].median())
    df_selected = df[['Pregnancies', 'Glucose', 'SkinThickness', 'BMI', 'Age', 'Outcome']]
    quantile_transformer = QuantileTransformer()
    df_transformed = pd.DataFrame(quantile_transformer.fit_transform(df_selected), columns=df_selected.columns)
    logging.info("Data preprocessing completed.")
    return df_transformed

def save_and_upload_data(df, datastore, output_file):
    """Saves the DataFrame to a CSV file and uploads it to the Azure ML datastore."""

    #full_path = os.path.join(path, file_name)
    #output_path = os.path.join("tmp/", output_file)

    #Exporting the file
    path = "tmp/"
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of directory %s failed" % path)
    else:
        print("Sucessfully created the directory %s " % path)

        
    temp_path = "tmp/" + output_file
    df.to_csv(temp_path)
    
    datastore.upload(src_dir="tmp/", target_path='', overwrite=True)
    
    logging.info(f"Data saved and uploaded to datastore: {temp_path}")


def main(input_file, output_file):
    config = load_config()
    ws = initialize_workspace(config)

    datastore = get_datastore(ws, "workspaceblobstore")

    df = load_data(datastore, input_file)
    df_processed = preprocess_data(df)
    save_and_upload_data(df_processed, datastore, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Complete data preprocessing workflow.")
    parser.add_argument("--input-file", type=str, required=True, help="Path to the input data file")
    parser.add_argument("--output-file", type=str, required=True, help="Path for saving the output CSV file")
    args = parser.parse_args()
    main( args.input_file, args.output_file)
