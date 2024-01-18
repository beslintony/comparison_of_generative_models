"""
Based on the given URL:
https://github.com/marload/GANs-TensorFlow2/blob/master/DCGAN/DCGAN.py
"""

import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras.callbacks as k
import matplotlib.pyplot as plt
import datetime
import os
import numpy as np
import argparse

from dt import load_dataset
from model_eval import ModelEvaluator

# Set environment variable
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Metrics setting
g_loss_metrics = tf.metrics.Mean(name='g_loss')
d_loss_metrics = tf.metrics.Mean(name='d_loss')
total_loss_metrics = tf.metrics.Mean(name='total_loss')

# Function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='DCGAN Training Script')
    parser.add_argument('--dataset', type=str, default='fashion_mnist', choices=['fashion_mnist', 'cifar10', 'svhn' ,'imagenet'], help='Dataset name')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--buffer_size', type=int, default=50000, help='Buffer size for dataset shuffling')
    parser.add_argument('--z_dim', type=int, default=100, help='Dimension of the generator input (z)')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--d_lr', type=float, default=0.0004, help='Discriminator learning rate')
    parser.add_argument('--g_lr', type=float, default=0.0004, help='Generator learning rate')
    parser.add_argument('--examples_to_generate', type=int, default=25, help='Number of examples to generate in each image')
    parser.add_argument('--save_image_freq', type=int, default=10, help='Frequency of saving generated images')
    parser.add_argument('--save_model_freq', type=int, default=100, help='Frequency of saving the generator model')
    parser.add_argument('--eval_freq', type=int, default=100, help='Frequency of printing evaluation metrics')
    parser.add_argument('--eval_batch_size', type=int, default=64, help='Batch size for evaluation metrics')
    parser.add_argument('--fid_gen_samples', type=int, default=10000, help='Number of generated samples for FID calculation')
    parser.add_argument('--fid_real_samples', type=int, default=10000, help='Number of real samples for FID calculation')
    parser.add_argument('--inception_score_samples', type=int, default=10000, help='Number of samples for Inception Score calculation')
    parser.add_argument('--wasserstein_distance_samples', type=int, default=10000, help='Number of samples for Wasserstein Distance calculation')

    return parser.parse_args()

# Get command-line arguments
args = parse_args()

# Test noise vector
test_z = tf.random.normal([36, args.z_dim])

class SaveImagesCallback(k.Callback):
    def __init__(self, model_name, dataset_name, decoder, latent_dim, examples_to_generate=25, save_freq=1, save_model_freq=10):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.decoder = decoder
        self.latent_dim = latent_dim
        self.examples_to_generate = examples_to_generate
        self.save_freq = save_freq
        self.save_model_freq = save_model_freq
        self.log_folder = self.create_log_folder()

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.save_freq == 0:
            latent_samples = np.random.normal(size=(self.examples_to_generate, self.latent_dim))
            generated_images = self.decoder.predict(latent_samples)
            generated_images = np.clip(generated_images, 0.0, 1.0)
            self.save_generated_images(generated_images, epoch)

        if epoch % self.save_model_freq == 0:
            self.save_model(epoch)

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
        
    def save_model(self, epoch):
        model_folder = os.path.join(self.log_folder, 'models')
        self.create_folder(model_folder)

        model_name = f'{self.model_name}_weights_epoch_{epoch}.h5'
        model_path = os.path.join(model_folder, model_name)
        self.decoder.save_weights(model_path)

    @staticmethod
    def create_folder(folder):
        if not os.path.exists(folder):
            os.makedirs(folder)

    def create_log_folder(self):
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_folder = os.path.join('logs', f'{self.model_name}_{self.dataset_name}_{current_time}')

        self.create_folder(log_folder)
        self.create_folder(os.path.join(log_folder, 'images'))
        self.create_folder(os.path.join(log_folder, 'models'))

        return log_folder

# Function to generate random noise vector
def get_random_z(z_dim, batch_size):
    return tf.random.uniform([batch_size, z_dim], minval=-1, maxval=1)

# Define discriminator
def make_discriminator(input_shape):
    return tf.keras.Sequential([
        layers.Conv2D(64, 5, strides=2, padding='same', input_shape=input_shape),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        layers.Conv2D(128, 5, strides=2, padding='same'),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(1)
    ])

# Define generator
def make_generator(input_shape):
    return tf.keras.Sequential([
        layers.Dense(8*8*256, use_bias=False, input_shape=input_shape),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Reshape((8, 8, 256)),
        layers.Conv2DTranspose(128, 5, strides=1, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Conv2DTranspose(64, 5, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Conv2DTranspose(3, 5, strides=2, padding='same', use_bias=False, activation='tanh')
    ])

# Define loss function
def get_loss_fn():
    criterion = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def d_loss_fn(real_logits, fake_logits):
        real_loss = criterion(tf.ones_like(real_logits), real_logits)
        fake_loss = criterion(tf.zeros_like(fake_logits), fake_logits)
        return real_loss + fake_loss

    def g_loss_fn(fake_logits):
        return criterion(tf.ones_like(fake_logits), fake_logits)

    return d_loss_fn, g_loss_fn

# Generator & Discriminator
G = make_generator((args.z_dim,))
D = make_discriminator((32, 32, 3))

# Optimizer
g_optim = tf.keras.optimizers.Adam(args.g_lr, beta_1=0.5, beta_2=0.999)
d_optim = tf.keras.optimizers.Adam(args.d_lr, beta_1=0.5, beta_2=0.999)

# Loss function
d_loss_fn, g_loss_fn = get_loss_fn()

@tf.function
def train_step(real_images):
    z = get_random_z(args.z_dim, args.batch_size)
    with tf.GradientTape() as d_tape, tf.GradientTape() as g_tape:
        fake_images = G(z, training=True)

        fake_logits = D(fake_images, training=True)
        real_logits = D(real_images, training=True)

        d_loss = d_loss_fn(real_logits, fake_logits)
        g_loss = g_loss_fn(fake_logits)

    d_gradients = d_tape.gradient(d_loss, D.trainable_variables)
    g_gradients = g_tape.gradient(g_loss, G.trainable_variables)

    d_optim.apply_gradients(zip(d_gradients, D.trainable_variables))
    g_optim.apply_gradients(zip(g_gradients, G.trainable_variables))

    return g_loss, d_loss

# Training loop
def train(ds, epochs=10, log_freq=20):
    ds_iter = iter(ds)  # Convert the dataset to an iterator
    evaluator = ModelEvaluator(batch_size=args.eval_batch_size)
    
    ds_for_evaluation, _ = load_dataset(args.dataset, buffer_size=args.buffer_size, batch_size=max(args.fid_real_samples, args.wasserstein_distance_samples), with_labels=False)
    real_images_for_evaluation = next(iter(ds_for_evaluation))
    real_images_array = real_images_for_evaluation.numpy()

    for epoch in range(epochs):
        for step in range(iterations_per_epoch):
            images = next(ds_iter)
            g_loss, d_loss = train_step(images)

            g_loss_metrics(g_loss)
            d_loss_metrics(d_loss)
            total_loss_metrics(g_loss + d_loss)

            if step % log_freq == 0:
                template = '[Epoch {}/{}] D_loss={:.5f} G_loss={:.5f} Total_loss={:.5f}'
                print(template.format(epoch + 1, epochs, d_loss_metrics.result(),
                                      g_loss_metrics.result(), total_loss_metrics.result()))
                g_loss_metrics.reset_states()
                d_loss_metrics.reset_states()
                total_loss_metrics.reset_states()
                
        # Evaluate and log at specified epochs
        if (epoch + 1) % args.eval_freq == 0:
            # Generate images for evaluation
            latent_samples = np.random.normal(size=(max(args.fid_gen_samples, args.inception_score_samples, args.wasserstein_distance_samples), args.latent_dim))
            generated_images_for_evaluation = G.predict(latent_samples)
            
            gen_images_array = generated_images_for_evaluation
            
            print("shape of real images: ", real_images_array.shape)
            print("shape of generated images: ", gen_images_array.shape)

            print(f'Calculating metrics...')
            is_avg, is_std = evaluator.calculate_inception_score(gen_images_array[:args.inception_score_samples])
            wasserstein_distance = evaluator.calculate_wasserstein_distance(real_images_array[:args.wasserstein_distance_samples],
                                                                             gen_images_array[:args.wasserstein_distance_samples])
            fid_score = evaluator.calculate_fid(real_images_array[:args.fid_real_samples],
                                                gen_images_array[:args.fid_gen_samples])
            print(f'Epoch {epoch + 1} - FID: {fid_score}, Inception Score: {is_avg:.4f} ± {is_std:.4f}, Wasserstein Distance: {wasserstein_distance}')

            # Log evaluation metrics to TensorBoard manually
            with tensorboard_writer.as_default():
                tf.summary.experimental.set_step(epoch + 1)
                tf.summary.scalar('fid_score', fid_score, step=epoch + 1)
                tf.summary.scalar('inception_score_avg', is_avg, step=epoch + 1)
                tf.summary.scalar('inception_score_std', is_std, step=epoch + 1)
                tf.summary.scalar('wasserstein_distance', wasserstein_distance, step=epoch + 1)

        # Log losses to TensorBoard manually
        with tensorboard_writer.as_default():
            tf.summary.experimental.set_step(epoch + 1)
            tf.summary.scalar('d_loss', d_loss_metrics.result(), step=epoch + 1)
            tf.summary.scalar('g_loss', g_loss_metrics.result(), step=epoch + 1)
            tf.summary.scalar('total_loss', total_loss_metrics.result(), step=epoch + 1)

        # Call the on_epoch_end method of the SaveImagesCallback
        save_images_callback.on_epoch_end(epoch + 1)

if __name__ == "__main__":
    # Calculate the number of iterations per epoch
    iterations_per_epoch = args.buffer_size // args.batch_size

    # Instantiate SaveImagesCallback
    save_images_callback = SaveImagesCallback(
        model_name='DCGAN',
        dataset_name=args.dataset,
        decoder=G,
        latent_dim=args.z_dim,
        examples_to_generate=args.examples_to_generate,
        save_freq=args.save_image_freq,
        save_model_freq=args.save_model_freq
    )
    
    tensorboard_writer = tf.summary.create_file_writer(save_images_callback.log_folder)

    # Load dataset
    train_ds, _ = load_dataset(dataset_name=args.dataset, buffer_size=args.buffer_size,
                            batch_size=args.batch_size, target_size=(32, 32), with_labels=False)

    # Train for the specified number of epochs
    train(train_ds, epochs=args.epochs)
