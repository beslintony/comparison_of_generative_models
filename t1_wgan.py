"""
Based on the given URL:
https://github.com/UestcJay/TensorFlow2-GAN/blob/master/implementations/wgan/wgan.py
"""

import argparse
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras import layers
import time
import datetime
from tensorflow.keras import callbacks as k

from dt import load_dataset
from model_eval import ModelEvaluator

# Set GPU device
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Metrics setting
g_loss_metrics = tf.metrics.Mean(name='g_loss')
d_loss_metrics = tf.metrics.Mean(name='d_loss')
total_loss_metrics = tf.metrics.Mean(name='total_loss')

def parse_arguments():
    parser = argparse.ArgumentParser(description='Wasserstein GAN Training Script')
    parser.add_argument('--dataset', type=str, default='fashion_mnist', choices=['fashion_mnist', 'cifar10', 'svhn' ,'imagenet'], help='Dataset name')
    parser.add_argument('--latent_dim', type=int, default=100, help='Size of the latent space')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for training')
    parser.add_argument('--d_lr', type=float, default=0.0004, help='Discriminator learning rate')
    parser.add_argument('--g_lr', type=float, default=0.0004, help='Generator learning rate')
    parser.add_argument('--beta_1', type=float, default=0.5, help='Beta_1 value for Adam optimizer')
    parser.add_argument('--beta_2', type=float, default=0.999, help='Beta_2 value for Adam optimizer')
    parser.add_argument('--dropout_rate', type=float, default=0.3, help='Dropout rate in the discriminator')
    parser.add_argument('--epochs', type=int, default=10000, help='Number of training epochs')
    parser.add_argument('--clip_val', type=int, default=0.01, help='Value for weight clipping')
    parser.add_argument('--buffer_size', type=int, default=50000, help='Buffer size for dataset shuffling')
    parser.add_argument('--examples_to_generate', type=int, default=25, help='Number of examples to generate in each image')
    parser.add_argument('--save_image_freq', type=int, default=1, help='Frequency of saving generated images')
    parser.add_argument('--save_model_freq', type=int, default=10, help='Frequency of saving the generator model')
    parser.add_argument('--eval_freq', type=int, default=1, help='Frequency of printing evaluation metrics')
    parser.add_argument('--eval_batch_size', type=int, default=64, help='Batch size for evaluation metrics')
    parser.add_argument('--fid_gen_samples', type=int, default=10000, help='Number of generated samples for FID calculation')
    parser.add_argument('--fid_real_samples', type=int, default=10000, help='Number of real samples for FID calculation')
    parser.add_argument('--inception_score_samples', type=int, default=10000, help='Number of samples for Inception Score calculation')
    parser.add_argument('--wasserstein_distance_samples', type=int, default=10000, help='Number of samples for Wasserstein Distance calculation')

    return parser.parse_args()

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

def make_generator(latent_dim):
    return tf.keras.Sequential([
        layers.Dense(8*8*256, use_bias=False, input_shape=(latent_dim,)),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Reshape((8, 8, 256)),
        layers.Conv2DTranspose(128, 5, strides=1, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Conv2DTranspose(64, 5, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Conv2DTranspose(3, 5, strides=2, padding='same', use_bias=False, activation='sigmoid')
    ])

def make_discriminator(img_shape, dropout_rate):
    return tf.keras.Sequential([
        layers.Conv2D(64, 5, strides=2, padding='same', input_shape=img_shape),
        layers.LeakyReLU(),
        layers.Dropout(dropout_rate),
        layers.Conv2D(128, 5, strides=2, padding='same'),
        layers.LeakyReLU(),
        layers.Dropout(dropout_rate),
        layers.Flatten(),
        layers.Dense(1)
    ])

def get_random_z(latent_dim, batch_size):
    return tf.random.uniform([batch_size, latent_dim], minval=-1, maxval=1)

def generator_loss(fake_output):
    return -tf.reduce_mean(fake_output)

def discriminator_loss(real_output, fake_output):
    return tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)

def train_step(images, generator, discriminator, generator_optimizer, discriminator_optimizer, latent_dim):
    noise = get_random_z(latent_dim, images.shape[0])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    
    for idx, grad in enumerate(gradients_of_discriminator):
        gradients_of_discriminator[idx] = tf.clip_by_value(grad, -args.clip_val, args.clip_val)
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
    return gen_loss, disc_loss

def train(generator, discriminator, generator_optimizer, discriminator_optimizer, train_dataset, args):
    evaluator = ModelEvaluator(batch_size=args.eval_batch_size)
    
    steps_per_epoch = args.buffer_size // args.batch_size

    ds_for_evaluation, _ = load_dataset(args.dataset, buffer_size=args.buffer_size, batch_size=max(args.fid_real_samples, args.wasserstein_distance_samples), with_labels=False)
    real_images_for_evaluation = next(iter(ds_for_evaluation))
    real_images_array = real_images_for_evaluation.numpy()

    for epoch in range(args.epochs):
                
        # Reset the metrics for each epoch
        g_loss_metrics.reset_states()
        d_loss_metrics.reset_states()
        total_loss_metrics.reset_states()

        for batch_idx, image_batch in enumerate(train_dataset.take(steps_per_epoch)):
            g_loss, d_loss = train_step(image_batch, generator, discriminator, generator_optimizer, discriminator_optimizer, args.latent_dim)

            g_loss_metrics(g_loss)
            d_loss_metrics(d_loss)
            total_loss_metrics(g_loss + d_loss)
            
            template = '[Epoch {}/{}], Batch [{}/{}] D_loss={:.5f} G_loss={:.5f} Total_loss={:.5f}'
            print(template.format(epoch + 1, args.epochs, batch_idx + 1, steps_per_epoch, d_loss_metrics.result(),
                                      g_loss_metrics.result(), total_loss_metrics.result()))

        # Callback for saving images and model
        save_callback.on_epoch_end(epoch + 1)
        
        # Evaluate and log at specified epochs
        if (epoch + 1) % args.eval_freq == 0:
            # Generate images for evaluation
            latent_samples = np.random.normal(size=(max(args.fid_gen_samples, args.inception_score_samples, args.wasserstein_distance_samples), args.latent_dim))
            generated_images_for_evaluation = generator.predict(latent_samples)
            
            gen_images_array = generated_images_for_evaluation
            gen_images_array = np.clip(gen_images_array, 0.0, 1.0)
            
            print("Real Image Min:", np.min(real_images_array))
            print("Real Image Max:", np.max(real_images_array))
            
            print("Gen Image Min:", np.min(gen_images_array))
            print("Gen Image Max:", np.max(gen_images_array))
            
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
        
if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()

    # Initialize generator and discriminator
    img_shape = (32, 32, 3)
    generator = make_generator(args.latent_dim)
    discriminator = make_discriminator(img_shape, args.dropout_rate)

    # Optimizers
    generator_optimizer = tf.keras.optimizers.Adam(args.g_lr, beta_1=args.beta_1, beta_2=args.beta_2)
    discriminator_optimizer = tf.keras.optimizers.Adam(args.g_lr, beta_1=args.beta_1, beta_2=args.beta_2)
    
    save_callback = SaveImagesCallback(
        model_name='WGAN',
        dataset_name=args.dataset,
        decoder=generator,
        latent_dim=args.latent_dim,
        examples_to_generate=args.examples_to_generate,
        save_freq=args.save_image_freq,
        save_model_freq=args.save_model_freq
    )

    tensorboard_writer = tf.summary.create_file_writer(save_callback.log_folder)
    
    # Log hyperparameters to TensorBoard with a common prefix
    with tensorboard_writer.as_default():
        tf.summary.scalar('hyperparameters/latent_dim', args.latent_dim, step=0)
        tf.summary.scalar('hyperparameters/d_lr', args.d_lr, step=0)
        tf.summary.scalar('hyperparameters/g_lr', args.g_lr, step=0)
        tf.summary.scalar('hyperparameters/beta_1', args.beta_1, step=0)
        tf.summary.scalar('hyperparameters/beta_2', args.beta_2, step=0)
        tf.summary.scalar('hyperparameters/dropout_rate', args.dropout_rate, step=0)
        tf.summary.scalar('hyperparameters/epochs', args.epochs, step=0)
        tf.summary.scalar('hyperparameters/batch_size', args.batch_size, step=0)
        tf.summary.scalar('hyperparameters/buffer_size', args.buffer_size, step=0)
        tf.summary.scalar('hyperparameters/clip_val', args.clip_val, step=0)
        tf.summary.scalar('hyperparameters/examples_to_generate', args.examples_to_generate, step=0)
        tf.summary.scalar('hyperparameters/save_image_freq', args.save_image_freq, step=0)
        tf.summary.scalar('hyperparameters/save_model_freq', args.save_model_freq, step=0)
        tf.summary.scalar('hyperparameters/eval_freq', args.eval_freq, step=0)
        tf.summary.scalar('hyperparameters/eval_batch_size', args.eval_batch_size, step=0)
        tf.summary.scalar('hyperparameters/fid_gen_samples', args.fid_gen_samples, step=0)
        tf.summary.scalar('hyperparameters/fid_real_samples', args.fid_real_samples, step=0)
        tf.summary.scalar('hyperparameters/inception_score_samples', args.inception_score_samples, step=0)
        tf.summary.scalar('hyperparameters/wasserstein_distance_samples', args.wasserstein_distance_samples, step=0)

    # Load dataset
    train_ds, _ = load_dataset(dataset_name=args.dataset, buffer_size=args.buffer_size,
                            batch_size=args.batch_size, target_size=(32, 32), with_labels=False)

    # Training loop
    train(generator, discriminator, generator_optimizer, discriminator_optimizer, train_ds, args)
