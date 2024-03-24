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

3. **Submit Experiments:**

   - Submit the generated experiments using the master script.

   ```bash
   bash submit_all.sh
   ```

4. **Interpreting Results:**
   - Examine the logs in `exps/logs` to analyze the performance progress of each experiment.
   - Compare metrics such as FID score, Inception score, and Wasserstein distance across different models and datasets.
   - To provide a streamlined experience, you could additionally add a custom folder to save the MLflow logs using the hyperparameter `base_log_folder`.

## Note

- Each experiment script specifies hyperparameters and dataset-specific configurations.
- Master script (`submit_all.sh`) submits all experiments for execution.
- Results can be analyzed to determine the effectiveness of different generative models on various datasets.
