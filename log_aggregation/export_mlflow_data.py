import os
import mlflow
import pandas as pd
import logging

# Set the tracking uri, update it with the correct one
mlflow.set_tracking_uri("http://localhost:5600")

# Set the tracking URI to your MLflow server
client = mlflow.MlflowClient(tracking_uri="http://localhost:5600")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def export_experiment_to_csv(experiment_id, all_data):
    # Get runs associated with the experiment
    runs = mlflow.search_runs(experiment_ids=[experiment_id])
    
    # Iterate through each run to aggregate data
    for _, run in runs.iterrows():
        # Get the run ID
        run_id = run['run_id']
        
        logger.info(f"Processing run {run_id}")
        
        # Get parameters for the run
        params = mlflow.get_run(run_id).data.params

        # Get all metrics' history for the run
        # Adapt it accordingly to the script and the metric names used
        metrics_history_fid = client.get_metric_history(run_id=run_id, key="FID Score")
        metrics_history_is_avg = client.get_metric_history(run_id=run_id, key="Avg. Inceprion Score")
        metrics_history_is_std = client.get_metric_history(run_id=run_id, key="Std. Inception Score")
        metrics_history_wd = client.get_metric_history(run_id=run_id, key="Wasserstein Distance")
        metrics_history_epoch = client.get_metric_history(run_id=run_id, key="Epoch")
        metrics_history_desc_loss = client.get_metric_history(run_id=run_id, key="Desc. Loss")
        metrics_history_gen_loss = client.get_metric_history(run_id=run_id, key="Gen. Loss")
        metrics_history_total_loss = client.get_metric_history(run_id=run_id, key="Total Loss")
        
        # for the column named exp_no, add a prefix DCGAN_ to the value
        params['exp_no'] = 'DCGAN_' + params['exp_no']
        # Add a new column named "model" with the value "DCGAN"
        params['model'] = 'DCGAN'

        # Extract metric values using epoch as step
        max_epoch = max([metric.step for metric in metrics_history_epoch]) if metrics_history_epoch else 0
        run_metrics_data = []
        for epoch in range(1, max_epoch + 1):  # Iterate through each epoch
            # Find metric values for this epoch
            fid_score = next((metric.value for metric in metrics_history_fid if metric.step == epoch), None)
            is_avg = next((metric.value for metric in metrics_history_is_avg if metric.step == epoch), None)
            is_std = next((metric.value for metric in metrics_history_is_std if metric.step == epoch), None)
            wd = next((metric.value for metric in metrics_history_wd if metric.step == epoch), None)
            desc_loss = next((metric.value for metric in metrics_history_desc_loss if metric.step == epoch), None)
            gen_loss = next((metric.value for metric in metrics_history_gen_loss if metric.step == epoch), None)
            total_loss = next((metric.value for metric in metrics_history_total_loss if metric.step == epoch), None)
            
            # Add row for this epoch with metric values and parameters
            row_data = {'run_id': run_id, 'epoch': epoch,
                        'FID Score': fid_score, 'Avg. Inception Score': is_avg, 'Std. Inception Score': is_std,
                        'Wasserstein Distance': wd, 'Desc. Loss': desc_loss, 'Gen. Loss': gen_loss,
                        'Total Loss': total_loss}
            row_data.update(params)  # Add parameters to row data
            run_metrics_data.append(row_data)
        
        # Extend the list with metric data for this run
        all_data.extend(run_metrics_data)
    
    logger.info(f"Finished processing all runs for experiment {experiment_id}")


# Get a list of all experiments
experiments = mlflow.search_experiments()

logger.info(f"Found {len(experiments)} experiments")

# Initialize an empty list to store all data
all_data = []

# Export data for each experiment
for experiment in experiments:
    experiment_name = experiment.name
    experiment_id = experiment.experiment_id
    
    logger.info(f"Exporting data for experiment {experiment_name} (ID: {experiment_id})")
    
    export_experiment_to_csv(experiment_id, all_data)

# Convert the list of dictionaries to a DataFrame
data_df = pd.DataFrame(all_data)

# Create a directory to store CSV files
output_dir = "combined_data"
os.makedirs(output_dir, exist_ok=True)

# Save DataFrame to CSV
csv_filename = os.path.join(output_dir, "DCGAN_combined.csv")
data_df.to_csv(csv_filename, index=False)
logger.info(f"All experiment data exported to {csv_filename}")
