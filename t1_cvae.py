"""
Based on the given URL:
https://github.com/s-omranpour/X-VAE-keras/blob/master/VAE/conditional%20VAE.ipynb
"""
import argparse
import os
import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras as k
from tensorflow.keras import layers
import matplotlib.pyplot as plt

from dt import load_dataset, load_validation_dataset
from model_eval import ModelEvaluator

# Define constants
LATENT_DIM = 10
LEARNING_RATE = 1e-4
BATCH_SIZE = 128
EPOCHS = 50
CONDITION_SIZE = 10
INPUT_SHAPE = (32, 32, 3)
EXAMPLES_TO_GENERATE = 25
SAVE_IMAGE_FREQ = 5
SAVE_MODEL_FREQ = 10
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


class SaveImagesCallback(k.callbacks.Callback):
    def __init__(
    self,
    model_name,
    dataset_name,
    decoder,
    latent_dim,
    condition_size,
    examples_to_generate=10,
    save_freq=1,
    save_model_freq=10,
    eval_freq=1,
    eval_batch_size=64,
    fid_gen_samples=10000,
    fid_real_samples=10000,
    inception_score_samples=10000,
    wasserstein_distance_samples=10000,
    ):
        # Initialize callback parameters
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.decoder = decoder
        self.latent_dim = latent_dim
        self.condition_size = condition_size
        self.examples_to_generate = examples_to_generate
        self.save_freq = save_freq
        self.save_model_freq = save_model_freq
        
        # Create log folder
        self.log_folder = self.create_log_folder()
        
        self.eval_freq = eval_freq
        self.eval_batch_size = eval_batch_size
        self.evaluator = ModelEvaluator(batch_size=self.eval_batch_size)
        self.fid_gen_samples = fid_gen_samples
        self.fid_real_samples = fid_real_samples
        self.inception_score_samples = inception_score_samples
        self.wasserstein_distance_samples = wasserstein_distance_samples

        self.tensorboard_writer = tf.summary.create_file_writer(self.log_folder)
        
        self.reconstruction_metric = tf.metrics.Mean(name='reconstruction_loss')
        self.kl_metric = tf.metrics.Mean(name='kl_loss')
        self.total_loss_metric = tf.metrics.Mean(name='total_loss')

    def on_epoch_end(self, epoch, logs=None):
        # Generate and save images
        if (epoch + 1) % self.save_freq == 0:
            latent_samples = np.random.normal(size=(self.examples_to_generate, self.latent_dim))
            condition_samples = np.random.randint(0, self.condition_size, size=(self.examples_to_generate, self.condition_size))
            generated_images = self.decoder.predict([latent_samples, condition_samples])
            generated_images = np.clip(generated_images, 0.0, 1.0)
            self.save_generated_images(generated_images, epoch + 1)

        # Save model
        if (epoch + 1) % self.save_model_freq == 0:
            self.save_model(epoch + 1)
        
        # Evaluate model
        if self.eval_freq != 0 and (epoch + 1) % self.eval_freq == 0:
            # Generate images for evaluation
            latent_samples = np.random.normal(
                size=(max(
                    args.fid_gen_samples, args.inception_score_samples, args.wasserstein_distance_samples),
                    args.latent_dim))

            condition_samples = np.random.randint(
                0, self.condition_size,
                size=(max(
                    args.fid_gen_samples, args.inception_score_samples, args.wasserstein_distance_samples),
                    self.condition_size))

            generated_images_for_evaluation = self.decoder.predict([latent_samples, condition_samples])
            gen_images_array = np.clip(generated_images_for_evaluation, 0.0, 1.0)

            # Load real images for evaluation
            ds_for_evaluation, _ = load_dataset(args.dataset, buffer_size=args.buffer_size,
                                                batch_size=max(args.fid_real_samples, args.wasserstein_distance_samples),
                                                with_labels=False)
            real_images_for_evaluation = next(iter(ds_for_evaluation))
            real_images_array = real_images_for_evaluation.numpy()
            
            print("Real Image Min:", np.min(real_images_array))
            print("Real Image Max:", np.max(real_images_array))
            
            print("Gen Image Min:", np.min(gen_images_array))
            print("Gen Image Max:", np.max(gen_images_array))
            
            # Calculate and print evaluation metrics
            is_avg, is_std = self.evaluator.calculate_inception_score(
                gen_images_array[:args.inception_score_samples])
            wasserstein_distance = self.evaluator.calculate_wasserstein_distance(
                real_images_array[:args.wasserstein_distance_samples],
                gen_images_array[:args.wasserstein_distance_samples])
            fid_score = self.evaluator.calculate_fid(real_images_array[:args.fid_real_samples],
                                                     gen_images_array[:args.fid_gen_samples])
            print(
                f'Epoch {epoch + 1} - FID: {fid_score}, Inception Score: {is_avg:.4f} ± {is_std:.4f}, Wasserstein Distance: {wasserstein_distance}')

            # Log evaluation metrics to TensorBoard manually
            with self.tensorboard_writer.as_default():
                tf.summary.experimental.set_step(epoch + 1)
                tf.summary.scalar('fid_score', fid_score, step=epoch + 1)
                tf.summary.scalar('inception_score_avg', is_avg, step=epoch + 1)
                tf.summary.scalar('inception_score_std', is_std, step=epoch + 1)
                tf.summary.scalar('wasserstein_distance', wasserstein_distance, step=epoch + 1)
        
        # Reset metrics for the next epoch
        self.reconstruction_metric.reset_states()
        self.kl_metric.reset_states()
        self.total_loss_metric.reset_states()

    def save_generated_images(self, generated_images, epoch):
        # Save generated images
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
        # Save model weights
        model_folder = os.path.join(self.log_folder, 'models')
        self.create_folder(model_folder)

        model_name = f'{self.model_name}_weights_epoch_{epoch}.h5'
        model_path = os.path.join(model_folder, model_name)
        self.decoder.save_weights(model_path)

    @staticmethod
    def create_folder(folder):
        # Create folder if it doesn't exist
        if not os.path.exists(folder):
            os.makedirs(folder)

    def create_log_folder(self):
        # Create a log folder based on timestamp
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_folder = os.path.join('logs', f'{self.model_name}_{self.dataset_name}_{current_time}')

        self.create_folder(log_folder)
        self.create_folder(os.path.join(log_folder, 'images'))
        self.create_folder(os.path.join(log_folder, 'models'))

        return log_folder

def load_dataset_samples(dataset_name, batch_size=128, buffer_size=60000):
    # Load the dataset
    dataset, num_classes = load_dataset(dataset_name, batch_size=batch_size, buffer_size=buffer_size, with_labels=True)
    # One-hot encode class labels
    dataset = dataset.map(lambda x, y: ((x, tf.one_hot(y, num_classes)), x))

    return dataset, num_classes

def load_val_dataset_samples(dataset_name, batch_size=128):
    # Load the dataset
    dataset, num_classes = load_validation_dataset(dataset_name, batch_size=batch_size)
    # One-hot encode class labels
    dataset = dataset.map(lambda x, y: ((x, tf.one_hot(y, num_classes)), x))

    return dataset

def make_encoder(input_shape=(32, 32, 3), latent_dim=64, condition_size=10):
    x = layers.Input(shape=input_shape)
    c = layers.Input(shape=(condition_size,))

    c_reshaped = layers.Dense(units=input_shape[0]*input_shape[1]*input_shape[2])(c)
    c_reshaped = layers.Reshape((input_shape[0], input_shape[1], input_shape[2]))(c_reshaped)

    h = layers.concatenate([x, c_reshaped])

    h = layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), padding='SAME', activation='relu')(h)
    h = layers.BatchNormalization()(h)
    h = layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), padding='SAME', activation='relu')(h)
    h = layers.BatchNormalization()(h)
    h = layers.Conv2D(filters=128, kernel_size=3, strides=(2, 2), padding='SAME', activation='relu')(h)
    h = layers.BatchNormalization()(h)
    h = layers.Flatten()(h)
    h = layers.Dense(256, activation='relu')(h)

    mean = layers.Dense(units=latent_dim)(h)
    log_var = layers.Dense(units=latent_dim)(h)

    return k.Model(inputs=[x, c], outputs=[mean, log_var], name='encoder')


def make_decoder(output_shape=(32, 32, 3), latent_dim=64, condition_size=10):
    z = layers.Input(shape=(latent_dim,))
    c = layers.Input(shape=(condition_size,))

    h = layers.concatenate([z, c])

    h = layers.Dense(units=4 * 4 * 128, activation='relu')(h)
    h = layers.Reshape((4, 4, 128))(h)
    h = layers.Conv2DTranspose(filters=64, kernel_size=4, strides=(2, 2), padding='same', activation='relu')(h)
    h = layers.BatchNormalization()(h)
    h = layers.Conv2DTranspose(filters=32, kernel_size=4, strides=(2, 2), padding='same', activation='relu')(h)
    h = layers.BatchNormalization()(h)

    y = layers.Conv2DTranspose(filters=output_shape[-1], kernel_size=4, strides=(2, 2), activation='sigmoid', padding='same')(h)
    
    return k.Model(inputs=[z, c], outputs=y, name='decoder')


def make_cvae_model(latent_dim=64, condition_size=10, learning_rate=0.001, input_shape=(32, 32, 3), alpha=1, beta=1):
    encoder = make_encoder(input_shape, latent_dim, condition_size)
    decoder = make_decoder(input_shape, latent_dim, condition_size)

    def sampling(args):
        mean, log_var = args
        batch = tf.shape(mean)[0]
        dim = tf.shape(mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return mean + tf.exp(0.5 * log_var) * epsilon

    x = layers.Input(shape=input_shape)
    c = layers.Input(shape=(condition_size,))

    mean, log_var = encoder([x, c])
    z = layers.Lambda(sampling, output_shape=(latent_dim,), name='sampling')([mean, log_var])
    y = decoder([z, c])

    reconstruction_loss = k.losses.mean_squared_error(y_true=x, y_pred=y)
    reconstruction_loss *= input_shape[0] * input_shape[1] * input_shape[2]

    kl_loss = -0.5 * tf.reduce_sum(1 + log_var - tf.square(mean) - tf.exp(log_var), axis=-1)
    kl_loss = tf.reduce_mean(kl_loss)

    cvae_loss = tf.reduce_mean(alpha * reconstruction_loss + beta * kl_loss)

    cvae = k.Model(inputs=[x, c], outputs=y, name='cvae')

    cvae.add_metric(tf.reduce_mean(reconstruction_loss), name='reconstruction_loss')
    cvae.add_metric(tf.reduce_mean(kl_loss), name='kl_loss')
    cvae.add_metric(tf.reduce_mean(cvae_loss), name='total_loss')

    cvae.add_loss(cvae_loss)

    cvae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

    return encoder, decoder, cvae

def main(args):
    train_dataset, num_classes = load_dataset_samples(args.dataset, batch_size=args.batch_size, buffer_size=args.buffer_size)
    val_dataset = load_val_dataset_samples(args.dataset, batch_size=args.batch_size)

    _, decoder, cvae = make_cvae_model(
        latent_dim=args.latent_dim,
        condition_size=num_classes,
        learning_rate=args.learning_rate
    )

    save_images_callback = SaveImagesCallback(
        model_name='t2-CVAE',
        dataset_name=args.dataset,
        decoder=decoder,
        latent_dim=args.latent_dim,
        condition_size=num_classes,
        examples_to_generate=args.examples_to_generate,
        save_freq=args.save_image_freq,
        save_model_freq=args.save_model_freq,
        eval_freq=args.eval_freq,
        eval_batch_size=args.eval_batch_size,
        fid_gen_samples=args.fid_gen_samples,
        fid_real_samples=args.fid_real_samples,
        inception_score_samples=args.inception_score_samples,
        wasserstein_distance_samples=args.wasserstein_distance_samples,
    )
    
    tensorboard_callback = k.callbacks.TensorBoard(log_dir=save_images_callback.log_folder)

    callbacks = [save_images_callback, tensorboard_callback]
    
    # Log hyperparameters to TensorBoard
    with save_images_callback.tensorboard_writer.as_default():
        tf.summary.scalar('hyperparameters/latent_dim', args.latent_dim, step=0)
        tf.summary.scalar('hyperparameters/learning_rate', args.learning_rate, step=0)
        tf.summary.scalar('hyperparameters/epochs', args.epochs, step=0)
        tf.summary.scalar('hyperparameters/batch_size', args.batch_size, step=0)
        tf.summary.scalar('hyperparameters/buffer_size', args.buffer_size, step=0)
        tf.summary.scalar('hyperparameters/examples_to_generate', args.examples_to_generate, step=0)
        tf.summary.scalar('hyperparameters/save_image_freq', args.save_image_freq, step=0)
        tf.summary.scalar('hyperparameters/save_model_freq', args.save_model_freq, step=0)
        tf.summary.scalar('hyperparameters/eval_freq', args.eval_freq, step=0)
        tf.summary.scalar('hyperparameters/eval_batch_size', args.eval_batch_size, step=0)
        tf.summary.scalar('hyperparameters/fid_gen_samples', args.fid_gen_samples, step=0)
        tf.summary.scalar('hyperparameters/fid_real_samples', args.fid_real_samples, step=0)
        tf.summary.scalar('hyperparameters/inception_score_samples', args.inception_score_samples, step=0)
        tf.summary.scalar('hyperparameters/wasserstein_distance_samples', args.wasserstein_distance_samples, step=0)

    cvae.fit(train_dataset, epochs=args.epochs, callbacks=callbacks, steps_per_epoch=args.buffer_size // args.batch_size, validation_data=val_dataset, validation_steps=10000 // args.batch_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train CVAE on different datasets')
    parser.add_argument('--dataset', type=str, default='fashion_mnist', choices=['fashion_mnist', 'cifar10', 'svhn' ,'imagenet'], help='Dataset name')
    parser.add_argument('--latent_dim', type=int, default=150, help='Latent dimension')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50000, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
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

    args = parser.parse_args()
    
    main(args)
