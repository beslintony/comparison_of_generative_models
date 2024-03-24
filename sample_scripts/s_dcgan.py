import os
import itertools
import pandas as pd

# Define the ranges or values for each hyperparameter
hyperparameter_ranges = {
    "learning_rate": ["0.0002", "0.0004", "0.0005"],
    "latent_dim": ["100", "150", "200"],
    "epochs": ["100", "300", "500"],
}

# Define dataset-specific configurations and default values
dataset_configs = {
    "fashion_mnist": "--save_image_freq 20 --save_model_freq 50 --eval_freq 10",
    "cifar10": "--save_image_freq 20 --save_model_freq 50 --eval_freq 10",
    "svhn": "--save_image_freq 20 --save_model_freq 50 --eval_freq 10",
    "imagenet": "--save_image_freq 20 --save_model_freq 50 --eval_freq 10"
}

# Default values
default_values = "--batch_size 64 --buffer_size 10000 --eval_batch_size 32 --fid_gen_samples 10000 --fid_real_samples 10000 --inception_score_samples 10000 --wasserstein_distance_samples 10000"

# Function to generate a bash script for an experiment
def generate_bash_script(dataset, dataset_args, config_args, exp_no):
    script_name = f"DCGAN_{dataset}_exp_{exp_no}.sh"
    
    base_folder = f"/clusterhome/user/comparison_of_generative_models"
    sbatch_log_folder = f"{base_folder}/exps/logs"

    script_content = f"""#!/bin/bash
#SBATCH --job-name=DCGAN_{dataset}_exp_{exp_no}
#SBATCH --output={sbatch_log_folder}/DCGAN_{dataset}_exp_{exp_no}.out
#SBATCH --error={sbatch_log_folder}/DCGAN_{dataset}_exp_{exp_no}.err

export BASE={base_folder}
export SCRIPT_PATH=${{BASE}}/t1_dcgan.py

singularity exec --nv ${{BASE}}/exps/ml_container.sif python ${{SCRIPT_PATH}} --dataset {dataset} --exp_no {exp_no} {dataset_args} {' '.join([f"--{param} {value}" for param, value in config_args.items()])}
"""
    with open(script_name, "w") as script_file:
        script_file.write(script_content)

# Function to generate a master bash script
def generate_master_script():
    master_content = "#!/bin/bash\n"
    for dataset, dataset_args in dataset_configs.items():
        dataset_args += " " + default_values
        exp_no = 1
        for _ in itertools.product(*hyperparameter_ranges.values()):
            script_name = f"DCGAN_{dataset}_exp_{exp_no}.sh"
            master_content += f"sbatch {script_name}\n"
            exp_no += 1
    
    with open("submit_all.sh", "w") as master_file:
        master_file.write(master_content)
        
# Function to generate experiment names and parameters
def generate_experiment_params():
    experiment_params = []
    for dataset, dataset_args in dataset_configs.items():
        dataset_args += " " + default_values
        for config_values in itertools.product(*hyperparameter_ranges.values()):
            config_args = dict(zip(hyperparameter_ranges.keys(), config_values))
            experiment_name = f"DCGAN_{dataset}_{'_'.join([f'{param}_{value}' for param, value in config_args.items()])}"
            experiment_params.append({"Experiment Name": experiment_name, **config_args})
    return experiment_params

# Generate individual experiment bash scripts
for dataset, dataset_args in dataset_configs.items():
    dataset_args += " " + default_values
    exp_no = 1
    for config_values in itertools.product(*hyperparameter_ranges.values()):
        config_args = dict(zip(hyperparameter_ranges.keys(), config_values))
        generate_bash_script(dataset, dataset_args, config_args, exp_no)
        exp_no += 1

# Generate the master bash script
generate_master_script()

# Generate experiment parameters
experiment_params = generate_experiment_params()

# Create a DataFrame from experiment parameters
experiment_params_df = pd.DataFrame(experiment_params)

# Save DataFrame to a CSV file
csv_filename = "experiment_parameters.csv"
experiment_params_df.to_csv(csv_filename, index=False)
print(f"Experiment parameters exported to {csv_filename}")