import mlflow
import os

def setup_mlflow_experiment(experiment_name, tracking_uri=None):
    """
    Sets up the MLflow experiment.
    
    Args:
        experiment_name (str): Name of the experiment.
        tracking_uri (str, optional): MLflow tracking URI. Defaults to 'mlruns' directory locally if not set.
        
    Returns:
        experiment_id (str): The ID of the active experiment.
    """
    if tracking_uri:
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

