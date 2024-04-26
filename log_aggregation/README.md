**MLflow Experiment Data Export Script**

### Overview

This repository contains a Python script for exporting experiment data from MLflow to a CSV file. The script aggregates data from MLflow tracking server and saves it into a structured CSV format for further analysis and visualization.

### Prerequisites

- Python 3.x
- MLflow installed (`pip install mlflow`)
- MLflow tracking server running

### Getting Started

1. **Start MLflow Tracking Server:**
   Ensure that MLflow tracking server is running and accessible. If not already installed, you can install it via pip:

   ```
   pip install mlflow
   ```

   Then start the MLflow server from the log data folder using:

   ```
   mlflow ui --port 5600
   ```

   **Note: If the folder path is not properly used, no data will be shown in the localhost server when the MLflow runs. So please start MLflow from the correct folder**.

2. **Run Script:**
   Execute the Python script `export_mlflow_data.py` to export experiment data to a CSV file:

   ```
   python export_mlflow_data.py
   ```

3. **Access Exported Data:**
   Once the script completes execution, you will find the exported CSV file named `MODEL_NAME_combined.csv` in the combined_data directory. Ensure to replace `MODEL_NAME` with the specific model name you are using.

### Customization

- **Script Modification:**
  You can modify the script `export_mlflow_data.py` according to your specific requirements, such as changing metric names, adding/removing parameters, or adjusting file output settings.

- **Exported CSV Format:**
  The exported CSV file contains aggregated experiment data with metrics and parameters. You can customize the structure of the CSV file by modifying the script accordingly.
