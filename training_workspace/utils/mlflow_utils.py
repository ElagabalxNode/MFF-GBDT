import mlflow
import os

def setup_mlflow_experiment(experiment_name, tracking_uri=None):
    """
    Sets up the MLflow experiment.
    
    Args:
        experiment_name (str): Name of the experiment.
        tracking_uri (str, optional): MLflow tracking URI. 
            If None, defaults to 'training_workspace/mlruns' directory.
        
    Returns:
        experiment_id (str): The ID of the active experiment.
    """
    # Set default tracking URI to training_workspace/mlruns if not provided
    if tracking_uri is None:
        # Use relative path to avoid Windows absolute path issues with MLflow
        # MLflow interprets Windows paths like "G:\..." as URI schemes
        tracking_uri = "training_workspace/mlruns"
        print(f"Using default MLflow tracking URI: {tracking_uri}")
    
    mlflow.set_tracking_uri(tracking_uri)
    
    # Check if experiment exists
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"Created new experiment: {experiment_name} (ID: {experiment_id})")
    else:
        experiment_id = experiment.experiment_id
        print(f"Using existing experiment: {experiment_name} (ID: {experiment_id})")
        
    mlflow.set_experiment(experiment_name)
    return experiment_id

def log_params_from_args(args, exclude=None):
    """
    Logs arguments from argparse Namespace as parameters.
    """
    if exclude is None:
        exclude = []
    
    params = vars(args)
    for key, value in params.items():
        if key not in exclude:
            mlflow.log_param(key, value)

