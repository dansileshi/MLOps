import os
import argparse
import logging
import yaml
from azureml.core import Workspace, Datastore, Dataset, Run
from azureml.core.authentication import InteractiveLoginAuthentication

# Set up basic configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

def get_workspace(config):
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

def get_datastore(workspace, datastore_name):
    """Fetches the Datastore from the Azure ML Workspace."""
    datastore = Datastore.get(workspace, datastore_name)
    logging.info(f"Retrieved datastore: {datastore.name}")
    return datastore

def read_dataset(datastore, file_path):
    """Reads a dataset from the specified datastore and path, and converts it to a pandas DataFrame.

    Args:
        datastore (Datastore): The Azure ML Datastore object.
        file_path (str): The path to the data file within the datastore.

    Returns:
        DataFrame: The loaded data as a pandas DataFrame.
    """
    df = Dataset.Tabular.from_delimited_files(path=[(datastore, file_path)]).to_pandas_dataframe()
    logging.info(f"Shape of Dataframe: {df.shape}")
    return df

def export_data(df, path, file_name):
    """Exports a DataFrame to a CSV file in the specified directory.

    Args:
        df (DataFrame): Pandas DataFrame to be exported.
        path (str): Directory path to store the output CSV file.
        file_name (str): Name of the output CSV file.

    Returns:
        str: Full path to the exported CSV file.
    """
    os.makedirs(path, exist_ok=True)
    full_path = os.path.join(path, file_name)
    df.to_csv(full_path)
    logging.info(f"Data exported to {full_path}")
    return full_path

def upload_to_datastore(datastore, src_dir, target_path=""):
    """Uploads a file from a specified directory to the Azure ML Datastore.

    Args:
        datastore (Datastore): The Azure ML Datastore object.
        src_dir (str): The source directory containing files to upload.
        target_path (str): The target path in the datastore where files will be uploaded.
    """
    datastore.upload(src_dir=src_dir, target_path=target_path, overwrite=True)
    logging.info("Data uploaded to datastore")

def main(input_data):
    """Main function to orchestrate the data processing workflow using Azure ML.

    Args:
        input_data (str): The input data file path for processing.
    """
    config = load_config()
    ws = get_workspace(config)
    datastore = get_datastore(ws, 'workspaceblobstore')
    df = read_dataset(datastore, input_data)
    file_path = export_data(df, "tmp/", "wrangled.csv")
    upload_to_datastore(datastore, "tmp/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and upload data using Azure ML")
    parser.add_argument("--input-data", type=str, required=True, help="Input data file path")
    args = parser.parse_args()
    main(args.input_data)
