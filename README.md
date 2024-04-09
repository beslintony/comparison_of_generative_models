# Comparison of Generative Models

This repository contains scripts for generating experiments to compare various generative models including CGAN, DCGAN, WGAN, WGAN-GP, VAE, and CVAE. The experiments are conducted on different datasets such as Fashion MNIST, CIFAR-10, SVHN, and ImageNet. The aim is to evaluate the performance of these models across different hyperparameters using evaluation metrics like Inception Score, FID, Wasserstein Distance.

## Usage

**Clone the Repository:**

```bash
git clone https://github.com/beslintony/comparison_of_generative_models.git
cd comparison_of_generative_models
```

**Setting Up the ImageNet Dataset:**

- Refer `preprocess/README.md`

**Generate Experiment Scripts:**

- Run the script generator for the desired model (CGAN, DCGAN, WGAN, WGAN-GP, VAE, CVAE) and dataset.
- For example, to generate CGAN experiments:

  ```bash
  cd sample_scripts
  python s_cgan.py
  ```

- Adjust the values in `hyperparameter_ranges`, `dataset_configs` or `default_values` inorder to change the hyperparameters across the generative models
- Replace the `base_folder` if there is a change in the root folder. NOTE: Better to use full path here.
- The singularity container image is expected to be found in the `exps` folder. The container image is called by default `ml_container.sif`. You could adapt the name here in their respective experiment generation script for the model.

**Submit Experiments:**

   - Submit the generated experiments using the master script.

   ```bash
   bash submit_all.sh
   ```

**Interpreting Results:**
   - Examine the logs in `exps/logs` to analyze the performance progress of each experiment.
   - Compare metrics such as FID score, Inception score, and Wasserstein distance across different models and datasets.
   - To provide a streamlined experience, you could additionally add a custom folder to save the MLflow logs using the hyperparameter `base_log_folder`.

#### Note

- Each experiment script specifies hyperparameters and dataset-specific configurations.
- Master script (`submit_all.sh`) submits all experiments for execution.
- Results can be analyzed to determine the effectiveness of different generative models on various datasets.

## Using Singularity Container

The experiments rely on a Singularity container image (ml_container.sif) to ensure reproducibility and portability. If you're not familiar with Singularity, you can refer to the [official Singularity documentation](https://docs.sylabs.io/guides/3.3/user-guide/quick_start.html).

**Creating the Singularity Image**

- You can create the `ml_container.sif` Singularity image by following these steps:
- Install Singularity on your system (v3.3).
- Pull the TensorFlow GPU Docker image and convert it to Singularity format:
   ```bash
   singularity build ml_container.sif docker://tensorflow/tensorflow:latest-gpu
   ```
**Adding Additional Python Modules**

If you need to add additional Python modules to the Singularity image, you can do so by creating a definition file (ml_container.def) and building the image with those dependencies. For example, to add the mlflow package:

- First, you need to start a shell session inside the Singularity container. You can do this using the following command.

```bash
singularity shell ml_container.sif
```

- Once inside the Singularity container's shell, you can use package managers like apt-get (for Debian-based distributions) or yum (for Red Hat-based distributions) to install new packages.

```bash
pip install mlflow matplotlib numpy tensorflow-datasets tensorflow-addons tensorflow-gan tensorflow-probability POT seaborn pandas tensorflow[and-cuda]
```
****Note: `pip install tensorflow[and-cuda]` helps to fix any cuda related errors and driver problems if you had any. For any other problems refer [Tensorflow documentations](https://www.tensorflow.org/install/pip).****

- After installing the desired packages, you can exit the shell session.

```bash
exit
```
