"""
Based on the given URL:
https://github.com/bnsreenu/python_for_microscopists/blob/master/249_keras_implementation-of_conditional_GAN/249-cifar_conditional_GAN.py
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
import mlflow

from dt import load_dataset
from model_eval import ModelEvaluator


# Set environment variable
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='CGAN Training Script')
    parser.add_argument('--dataset', type=str, default='fashion_mnist', choices=['fashion_mnist', 'cifar10', 'svhn' ,'imagenet'], help='Dataset name')
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--buffer_size', type=int, default=10000, help='Buffer size for dataset shuffling')
    parser.add_argument('--latent_dim', type=int, default=100, help='Size of the latent space')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--beta_1', type=float, default=0.5, help='Beta_1 value for Adam optimizer')
    parser.add_argument('--beta_2', type=float, default=0.999, help='Beta_2 value for Adam optimizer')
    parser.add_argument('--dropout_rate', type=float, default=0.3, help='Dropout rate in the discriminator')
    parser.add_argument('--examples_to_generate', type=int, default=25, help='Number of examples to generate in each image')
    parser.add_argument('--save_image_freq', type=int, default=20, help='Frequency of saving generated images')
    parser.add_argument('--save_model_freq', type=int, default=100, help='Frequency of saving the generator model')
    parser.add_argument('--eval_freq', type=int, default=10, help='Frequency of printing evaluation metrics')
    parser.add_argument('--eval_batch_size', type=int, default=32, help='Batch size for evaluation metrics')
    parser.add_argument('--fid_gen_samples', type=int, default=10000, help='Number of generated samples for FID calculation')
    parser.add_argument('--fid_real_samples', type=int, default=10000, help='Number of real samples for FID calculation')
    parser.add_argument('--inception_score_samples', type=int, default=10000, help='Number of samples for Inception Score calculation')
    parser.add_argument('--wasserstein_distance_samples', type=int, default=10000, help='Number of samples for Wasserstein Distance calculation')
    parser.add_argument('--exp_no', type=int, default=0, help='The experiment number')
    parser.add_argument('--base_log_folder', type=str, default='/tmp/logs', help='The experiment number')

    return parser.parse_args()

# Get command-line arguments
args = parse_args()

class SaveCallback(tf.keras.callbacks.Callback):
    def __init__(self, model_name, dataset_name, generator, latent_dim, num_classes, examples_to_generate=25, save_freq=1, save_model_freq=10, exp_no=0, base_log_folder='/tmp/logs'):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.generator = generator
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.examples_to_generate = examples_to_generate
        self.save_freq = save_freq
        self.save_model_freq = save_model_freq
        self.exp_no = exp_no
        self.base_log_folder = base_log_folder
        self.log_folder = self.create_log_folder()

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.save_freq == 0:
            latent_samples = np.random.normal(size=(self.examples_to_generate, self.latent_dim))
            labels = np.random.randint(0, self.num_classes, self.examples_to_generate)
            generated_images = self.generator.predict([latent_samples, labels])
            generated_images = np.clip(generated_images, 0.0, 1.0)
            folder_path = self.save_generated_images(generated_images, epoch)
            mlflow.log_artifact(folder_path)

        if epoch % self.save_model_freq == 0:
            self.save_model(epoch)

    def save_generated_images(self, generated_images, epoch):
        folder_path = os.path.join(self.log_folder, 'images')

        plt.figure(figsize=(10, 10))
        for i in range(self.examples_to_generate):
            plt.subplot(5, 5, i + 1)
            plt.imshow(generated_images[i])
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

        model_name = f'{self.model_name}_weights_epoch_{epoch}.h5'
        model_path = os.path.join(model_folder, model_name)
        self.generator.save_weights(model_path)

        # Log the model weights as artifacts in MLflow
        mlflow.log_artifact(model_path)

    @staticmethod
    def create_folder(folder):
        if not os.path.exists(folder):
            os.makedirs(folder)

    def create_log_folder(self):
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_folder = os.path.join(self.base_log_folder, self.model_name, self.dataset_name,
                                  f'{self.exp_no}_{current_time}')

        self.create_folder(log_folder)
        self.create_folder(os.path.join(log_folder, 'images'))
        self.create_folder(os.path.join(log_folder, 'models'))

        return log_folder

# Metrics setting
g_loss_metrics = tf.metrics.Mean(name='g_loss')
d_loss_metrics = tf.metrics.Mean(name='d_loss')
total_loss_metrics = tf.metrics.Mean(name='total_loss')

# Function to generate random noise vector
def get_random_z(latent_dim, batch_size):
    return tf.random.uniform([batch_size, latent_dim], minval=-1, maxval=1)

# Define discriminator
def make_discriminator(input_shape, num_classes, dropout_rate):
    input_image = layers.Input(shape=input_shape)
    input_label = layers.Input(shape=(), dtype=tf.int32)
    
    label_embedding = layers.Embedding(num_classes, np.prod(input_shape))(input_label)
    flat_embedding = layers.Flatten()(label_embedding)
    reshaped_embedding = layers.Reshape(input_shape)(flat_embedding)
    concatenated_input = layers.Concatenate()([input_image, reshaped_embedding])
    
    x = layers.Conv2D(64, 5, strides=2, padding='same')(concatenated_input)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Conv2D(128, 5, strides=2, padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(1)(x)
    
    return tf.keras.Model(inputs=[input_image, input_label], outputs=x)

# Define generator
def make_generator(latent_dim, num_classes):
    input_noise = layers.Input(shape=(latent_dim,))
    input_label = layers.Input(shape=(), dtype=tf.int32)
    
    label_embedding = layers.Embedding(num_classes, latent_dim)(input_label)
    merged_input = layers.Multiply()([input_noise, label_embedding])
    
    x = layers.Dense(8*8*256, use_bias=False)(merged_input)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Reshape((8, 8, 256))(x)
    
    x = layers.Conv2DTranspose(128, 5, strides=1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    
    x = layers.Conv2DTranspose(64, 5, strides=2, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    
    x = layers.Conv2DTranspose(3, 5, strides=2, padding='same', use_bias=False, activation='sigmoid')(x)
    
    return tf.keras.Model(inputs=[input_noise, input_label], outputs=x)

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


@tf.function
def train_step(real_images, labels):
    z = get_random_z(args.latent_dim, args.batch_size)
    with tf.GradientTape() as d_tape, tf.GradientTape() as g_tape:
        fake_images = G([z, labels], training=True)

        fake_logits = D([fake_images, labels], training=True)
        real_logits = D([real_images, labels], training=True)

        d_loss = d_loss_fn(real_logits, fake_logits)
        g_loss = g_loss_fn(fake_logits)

    d_gradients = d_tape.gradient(d_loss, D.trainable_variables)
    g_gradients = g_tape.gradient(g_loss, G.trainable_variables)

    d_optim.apply_gradients(zip(d_gradients, D.trainable_variables))
    g_optim.apply_gradients(zip(g_gradients, G.trainable_variables))

    return g_loss, d_loss

# Training loop
def train(ds, epochs=10, log_freq=20):
    ds_iter = iter(ds)
    evaluator = ModelEvaluator(batch_size=args.eval_batch_size)

    ds_for_evaluation, num_classes = load_dataset(args.dataset, buffer_size=args.buffer_size, batch_size=max(args.fid_real_samples, args.wasserstein_distance_samples), with_labels=True)
    real_images_for_evaluation, _ = next(iter(ds_for_evaluation))
    real_images_array = real_images_for_evaluation.numpy()

    for epoch in range(epochs):
        for step in range(iterations_per_epoch):
            images, labels = next(ds_iter)
            g_loss, d_loss = train_step(images, labels)

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
            images_to_generate = max(args.fid_gen_samples, args.inception_score_samples, args.wasserstein_distance_samples)
            # Generate images for evaluation
            latent_samples = np.random.normal(size=(images_to_generate, args.latent_dim))
            labels = np.random.randint(0, num_classes, images_to_generate)
            generated_images_for_evaluation = G.predict([latent_samples, labels])
            
            gen_images_array = generated_images_for_evaluation
            gen_images_array = np.clip(gen_images_array, 0.0, 1.0)
            
            print(f'Calculating metrics...')
            is_avg, is_std = evaluator.calculate_inception_score(gen_images_array[:args.inception_score_samples])
            wasserstein_distance = evaluator.calculate_wasserstein_distance(real_images_array[:args.wasserstein_distance_samples],
                                                                             gen_images_array[:args.wasserstein_distance_samples])
            fid_score = evaluator.calculate_fid(real_images_array[:args.fid_real_samples],
                                                gen_images_array[:args.fid_gen_samples])
            print(
                f'Epoch= {epoch + 1}, FID= {fid_score}, Inception Score= {is_avg:.4f} ± {is_std:.4f}, Wasserstein Distance= {wasserstein_distance}')

            mlflow.log_metric("FID Score", fid_score, step=epoch + 1)
            mlflow.log_metric("Avg. Inception Score", is_avg, step=epoch + 1)
            mlflow.log_metric("Std. Inception Score", is_std, step=epoch + 1)
            mlflow.log_metric("Wasserstein Distance", wasserstein_distance, step=epoch + 1)
            mlflow.log_metric("Epoch", epoch+1, step=epoch + 1)
            
        mlflow.log_metric("Desc. Loss", d_loss_metrics.result(), step=epoch + 1)
        mlflow.log_metric("Gen. Loss", g_loss_metrics.result(), step=epoch + 1)
        mlflow.log_metric("Total Loss", total_loss_metrics.result(), step=epoch + 1)

        # Call the on_epoch_end method of the SaveCallback
        save_callback.on_epoch_end(epoch + 1)
        
    # Save model weights at the end of training
    G.save_weights(os.path.join(save_callback.log_folder, 'final_generator_weights.h5'))

if __name__ == "__main__":
    # Initialize MLflow and create an experiment
    mlflow.set_experiment(f'CGAN_{args.dataset}_exp_{args.exp_no}')
    mlflow.start_run()
    mlflow.set_tags({"model": "CGAN", "dataset": args.dataset, "exp_no": args.exp_no})
    
    # Calculate the number of iterations per epoch
    iterations_per_epoch = args.buffer_size // args.batch_size
    
    # Load dataset with labels
    train_ds, num_classes = load_dataset(dataset_name=args.dataset, buffer_size=args.buffer_size,
                            batch_size=args.batch_size, target_size=(32, 32), with_labels=True)
    
    G = make_generator(args.latent_dim, num_classes)
    D = make_discriminator((32, 32, 3), num_classes, dropout_rate=args.dropout_rate)

    # Optimizer
    g_optim = tf.keras.optimizers.Adam(args.learning_rate, beta_1=args.beta_1, beta_2=args.beta_2)
    d_optim = tf.keras.optimizers.Adam(args.learning_rate, beta_1=args.beta_1, beta_2=args.beta_2)

    # Loss function
    d_loss_fn, g_loss_fn = get_loss_fn()

    save_callback = SaveCallback(
        model_name='CGAN',
        dataset_name=args.dataset,
        num_classes=num_classes,
        generator=G,
        latent_dim=args.latent_dim,
        examples_to_generate=25,
        save_freq=args.save_image_freq,
        save_model_freq=args.save_model_freq,
        exp_no=args.exp_no,
        base_log_folder=args.base_log_folder
    )
    
    # Log hyperparameters
    mlflow.log_param("latent_dim", args.latent_dim)
    mlflow.log_param("learning_rate", args.learning_rate)
    mlflow.log_param("epochs", args.epochs)
    mlflow.log_param("dataset", args.dataset)
    mlflow.log_param("exp_no", args.exp_no)

    # Train for the specified number of epochs
    train(train_ds, epochs=args.epochs)
    
    # End the MLflow run
    mlflow.end_run()