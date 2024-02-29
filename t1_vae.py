"""
Based on the given URL:
https://github.com/s-omranpour/X-VAE-keras/blob/master/VAE/VAE.py
https://keras.io/examples/generative/vae/
"""

import argparse
import os
import datetime

import tensorflow as tf
import numpy as np
from tensorflow import keras as k
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import mlflow

from dt import load_dataset, load_validation_dataset
from model_eval import ModelEvaluator

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
class SaveCallback(k.callbacks.Callback):
    def __init__(
        self,
        model_name,
        dataset_name,
        decoder,
        latent_dim,
        examples_to_generate=10,
        save_freq=1,
        save_model_freq=10,
        eval_freq=1,
        eval_batch_size=64,
        fid_gen_samples=10000,
        fid_real_samples=10000,
        inception_score_samples=10000,
        wasserstein_distance_samples=10000,
        exp_no=0,
        base_log_folder='/tmp/logs'
    ):

        self.model_name = model_name
        self.dataset_name = dataset_name
        self.decoder = decoder
        self.latent_dim = latent_dim
        self.examples_to_generate = examples_to_generate
        self.save_freq = save_freq
        self.save_model_freq = save_model_freq
        
        self.exp_no = exp_no
        self.base_log_folder = base_log_folder
        self.log_folder = self.create_log_folder()

        self.eval_freq = eval_freq
        self.eval_batch_size = eval_batch_size
        self.evaluator = ModelEvaluator(batch_size=self.eval_batch_size)
        self.fid_gen_samples = fid_gen_samples
        self.fid_real_samples = fid_real_samples
        self.inception_score_samples = inception_score_samples
        self.wasserstein_distance_samples = wasserstein_distance_samples
        
        # Loss metrics
        self.reconstruction_metric = tf.metrics.Mean(name='reconstruction_loss')
        self.kl_metric = tf.metrics.Mean(name='kl_loss')
        self.total_loss_metric = tf.metrics.Mean(name='total_loss')

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.save_freq == 0:
            latent_samples = np.random.normal(size=(self.examples_to_generate, self.latent_dim))
            generated_images = self.decoder.predict(latent_samples)
            generated_images = np.clip(generated_images, 0.0, 1.0)
            folder_path = self.save_generated_images(generated_images, epoch + 1)
            mlflow.log_artifact(folder_path)
        
        if (epoch + 1) % self.save_model_freq == 0:
            self.save_model(epoch + 1)

        if self.eval_freq != 0 and (epoch + 1) % self.eval_freq == 0:
            # Generate images for evaluation
            latent_samples = np.random.normal(
                size=(max(
                    args.fid_gen_samples, args.inception_score_samples, args.wasserstein_distance_samples),
                    args.latent_dim))
            generated_images_for_evaluation = self.decoder.predict(latent_samples)
            gen_images_array = generated_images_for_evaluation
            gen_images_array = np.clip(gen_images_array, 0.0, 1.0)
            
            # Load real images for evaluation
            ds_for_evaluation, _ = load_dataset(args.dataset, buffer_size=args.buffer_size,
                                               batch_size=max(args.fid_real_samples, args.wasserstein_distance_samples),
                                               with_labels=False)
            real_images_for_evaluation = next(iter(ds_for_evaluation))
            real_images_array = real_images_for_evaluation.numpy()

            # Calculate and print evaluation metrics
            is_avg, is_std = self.evaluator.calculate_inception_score(
                gen_images_array[:args.inception_score_samples])
            wasserstein_distance = self.evaluator.calculate_wasserstein_distance(
                real_images_array[:args.wasserstein_distance_samples],
                gen_images_array[:args.wasserstein_distance_samples])
            fid_score = self.evaluator.calculate_fid(real_images_array[:args.fid_real_samples],
                                                     gen_images_array[:args.fid_gen_samples])
            print(
                f'Epoch= {epoch + 1}, FID= {fid_score}, Inception Score= {is_avg:.4f} ± {is_std:.4f}, Wasserstein Distance= {wasserstein_distance}')

            mlflow.log_metric("FID Score", fid_score, step=epoch + 1)
            mlflow.log_metric("Avg. Inceprion Score", is_avg, step=epoch + 1)
            mlflow.log_metric("Std. Inception Score", is_std, step=epoch + 1)
            mlflow.log_metric("Wasserstein Distance", wasserstein_distance, step=epoch + 1)
            mlflow.log_metric("Epoch", epoch + 1, step=epoch + 1)
        
        mlflow.log_metric("Recon. Loss", self.reconstruction_metric.result(), step=epoch + 1)
        mlflow.log_metric("KL Loss", self.kl_metric.result(), step=epoch + 1)
        mlflow.log_metric("Total Loss", self.total_loss_metric.result(), step=epoch + 1)

        # Reset metrics for the next epoch
        self.reconstruction_metric.reset_states()
        self.kl_metric.reset_states()
        self.total_loss_metric.reset_states()

    def save_generated_images(self, generated_images, epoch):
        folder_path = os.path.join(self.log_folder, 'images')

        plt.figure(figsize=(10, 10))
        for i in range(self.examples_to_generate):
            plt.subplot(5, 5, i + 1)
            plt.imshow(generated_images[i, :, :, :])
            plt.axis('off')

        plt.tight_layout()

        self.create_folder(folder_path)

        file_name = f'{folder_path}/generated_images_epoch_{epoch}.png'
        plt.savefig(file_name)
        plt.close()
        
        return folder_path

    def save_model(self, epoch):
        model_folder = os.path.join(self.log_folder, 'models')
        self.create_folder(model_folder)

        model_name = f'{self.model_name}_epoch_{epoch}.h5'
        model_path = os.path.join(model_folder, model_name)
        self.decoder.save(model_path)
        
        # Log the model weights as artifacts in MLflow
        mlflow.log_artifact(model_path)

    @staticmethod
    def create_folder(folder):
        if not os.path.exists(folder):
            os.makedirs(folder)

    def create_log_folder(self):
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_folder = os.path.join(self.base_log_folder, self.model_name, self.dataset_name, f'{self.exp_no}_{current_time}')

        self.create_folder(log_folder)
        self.create_folder(os.path.join(log_folder, 'images'))
        self.create_folder(os.path.join(log_folder, 'models'))

        return log_folder

def make_model(SIZE=(32, 32, 3), LATENT_DIM=10, LR=1e-4, BETA=1.0):
    # Encoder
    encoder_inputs = layers.Input(shape=SIZE, name='encoder_input')
    e = layers.Conv2D(filters=32, kernel_size=4, strides=(2, 2), padding='SAME', activation='relu')(encoder_inputs)
    e = layers.BatchNormalization()(e)
    e = layers.Conv2D(filters=64, kernel_size=4, strides=(2, 2), padding='SAME', activation='relu')(e)
    e = layers.BatchNormalization()(e)
    e = layers.Conv2D(filters=128, kernel_size=4, strides=(2, 2), padding='SAME', activation='relu')(e)
    e = layers.BatchNormalization()(e)
    e = layers.Flatten()(e)
    e = layers.Dense(256, activation='relu')(e)
    
    z_mean = layers.Dense(LATENT_DIM, name='z_mean')(e)
    z_log_var = layers.Dense(LATENT_DIM, name='z_log_var')(e)
    encoder = k.Model(inputs=encoder_inputs, outputs=[z_mean, z_log_var], name='encoder')

    # Decoder
    decoder_inputs = layers.Input(shape=(LATENT_DIM,), name='decoder_input')
    d = layers.Dense(units=4 * 4 * 128, activation='relu')(decoder_inputs)
    d = layers.Reshape((4, 4, 128))(d)
    d = layers.Conv2DTranspose(filters=64, kernel_size=4, strides=(2, 2), padding="SAME", activation='relu')(d)
    d = layers.BatchNormalization()(d)
    d = layers.Conv2DTranspose(filters=32, kernel_size=4, strides=(2, 2), padding="SAME", activation='relu')(d)
    d = layers.BatchNormalization()(d)
    d = layers.Conv2DTranspose(filters=3, kernel_size=4, strides=(2, 2), padding="SAME", activation='sigmoid')(d)
    decoder = k.Model(inputs=decoder_inputs, outputs=d, name='decoder')
    
# Some modifications of the loss calculations from 
# https://github.com/s-omranpour/X-VAE-keras/blob/master/VAE/conditional%20VAE.ipynb
    
    # VAE
    def sample(inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    sampler = layers.Lambda(sample)
    z = sampler([z_mean, z_log_var])
    vae_outputs = decoder(z)
    vae = k.Model(inputs=encoder_inputs, outputs=vae_outputs, name='vae')

    # Loss calculation
    reconstruction_loss = k.losses.mean_squared_error(encoder_inputs, vae_outputs)
    reconstruction_loss *= SIZE[0] * SIZE[1] * SIZE[2]

    kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
    kl_loss = tf.reduce_mean(kl_loss)

    vae_loss = tf.reduce_mean(reconstruction_loss + BETA * kl_loss, name='vae_loss')
    
    # Update model to include loss metrics
    vae.add_metric(reconstruction_loss, name='reconstruction_loss')
    vae.add_metric(kl_loss, name='kl_loss')
    vae.add_metric(vae_loss, name='total_loss')

    vae.add_loss(vae_loss)
    vae.compile(optimizer=k.optimizers.Adam(LR))

    return encoder, decoder, vae

def main(args):
    train_dataset, _ = load_dataset(args.dataset, buffer_size=args.buffer_size, batch_size=args.batch_size,
                                    with_labels=False)
    val_dataset, _ = load_validation_dataset(args.dataset, batch_size=args.batch_size, with_labels=False)

    _, decoder, vae = make_model(SIZE=(32, 32, 3), LATENT_DIM=args.latent_dim, LR=args.learning_rate, BETA=args.beta)

    save_callback = SaveCallback(
        model_name='VAE',
        dataset_name=args.dataset,
        decoder=decoder,
        latent_dim=args.latent_dim,
        examples_to_generate=args.examples_to_generate,
        save_freq=args.save_image_freq,
        save_model_freq=args.save_model_freq,
        eval_freq=args.eval_freq,
        eval_batch_size=args.eval_batch_size,
        fid_gen_samples=args.fid_gen_samples,
        fid_real_samples=args.fid_real_samples,
        inception_score_samples=args.inception_score_samples,
        wasserstein_distance_samples=args.wasserstein_distance_samples,
        exp_no=args.exp_no,
        base_log_folder=args.base_log_folder
    )
    
    callbacks = [save_callback]

    vae.fit(
        train_dataset.map(lambda x: (x, x)),
        epochs=args.epochs,
        callbacks=callbacks,
        steps_per_epoch=args.buffer_size // args.batch_size,
        validation_data=val_dataset.map(lambda x: (x, x)),
        validation_steps=10000 // args.batch_size
    )
    
    # Save model weights at the end of training
    decoder.save_weights(os.path.join(save_callback.log_folder, 'final_generator_weights.h5'))    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train VAE on different datasets')
    parser.add_argument('--dataset', type=str, default='fashion_mnist',
                        choices=['fashion_mnist', 'cifar10', 'svhn', 'imagenet'], help='Dataset name')
    parser.add_argument('--latent_dim', type=int, default=150, help='Latent dimension')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--beta', type=float, default=1.0, help='Beta hyperparameter')
    parser.add_argument('--epochs', type=int, default=5000, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--buffer_size', type=int, default=60000, help='Buffer size for dataset shuffling')
    parser.add_argument('--examples_to_generate', type=int, default=25, help='Number of examples to generate in each image')
    parser.add_argument('--save_image_freq', type=int, default=1, help='Frequency of saving generated images')
    parser.add_argument('--save_model_freq', type=int, default=10, help='Frequency of saving the generator model')
    parser.add_argument('--eval_freq', type=int, default=1, help='Frequency of printing evaluation metrics')
    parser.add_argument('--eval_batch_size', type=int, default=64, help='Batch size for evaluation metrics')
    parser.add_argument('--fid_gen_samples', type=int, default=10000, help='Number of generated samples for FID calculation')
    parser.add_argument('--fid_real_samples', type=int, default=10000, help='Number of real samples for FID calculation')
    parser.add_argument('--inception_score_samples', type=int, default=10000, help='Number of samples for Inception Score calculation')
    parser.add_argument('--wasserstein_distance_samples', type=int, default=10000, help='Number of samples for Wasserstein Distance calculation')
    parser.add_argument('--exp_no', type=int, default=0, help='The experiment number')
    parser.add_argument('--base_log_folder', type=str, default='/tmp/logs', help='The experiment number')

    args = parser.parse_args()
    
    # Initialize MLflow and create an experiment
    mlflow.set_experiment(f'VAE_{args.dataset}_exp_{args.exp_no}')
    mlflow.start_run()
    mlflow.set_tags({"model": "VAE", "dataset": args.dataset, "exp_no": args.exp_no})
    
    # Log hyperparameters
    mlflow.log_param("latent_dim", args.latent_dim)
    mlflow.log_param("learning_rate", args.learning_rate)
    mlflow.log_param("beta", args.beta)
    mlflow.log_param("epochs", args.epochs)
    mlflow.log_param("dataset", args.dataset)
    mlflow.log_param("exp_no", args.exp_no)
    
    main(args)
    
    # End the MLflow run
    mlflow.end_run()
