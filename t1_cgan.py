"""
Based on the given URL:
https://github.com/bnsreenu/python_for_microscopists/blob/master/249_keras_implementation-of_conditional_GAN/249-cifar_conditional_GAN.py
"""
import argparse
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU, Dropout, Embedding, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import datetime
from tensorflow.keras import callbacks as k
import mlflow

from model_eval import ModelEvaluator
from dt import load_dataset

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Constants
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

class SaveCallback(k.Callback):
    def __init__(self, model_name, dataset_name, num_classes, generator, latent_dim, examples_to_generate=25, save_freq=1, save_model_freq=10, exp_no=0, base_log_folder='/tmp/logs'):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.num_classes = num_classes
        self.generator = generator
        self.latent_dim = latent_dim
        self.examples_to_generate = examples_to_generate
        self.save_freq = save_freq
        self.save_model_freq = save_model_freq
        self.exp_no = exp_no
        self.base_log_folder = base_log_folder
        self.log_folder = self.create_log_folder()

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.save_freq == 0:
            latent_samples, labels = generate_latent_points(self.latent_dim, self.examples_to_generate, n_classes=self.num_classes)
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
        log_folder = os.path.join(self.base_log_folder, self.model_name, self.dataset_name, f'{self.exp_no}_{current_time}')

        self.create_folder(log_folder)
        self.create_folder(os.path.join(log_folder, 'images'))
        self.create_folder(os.path.join(log_folder, 'models'))

        return log_folder

def define_discriminator(in_shape=(32, 32, 3), n_classes=10, dropout_rate=0.4, learning_rate=0.0002, beta_1=0.5, alpha=0.2, beta_2=0.999):
    # Label input
    in_label = Input(shape=(1,))
    li = Embedding(n_classes, 50)(in_label)
    n_nodes = in_shape[0] * in_shape[1]
    li = Dense(n_nodes)(li)
    li = Reshape((in_shape[0], in_shape[1], 1))(li)

    # Image input
    in_image = Input(shape=in_shape)
    merge = Concatenate()([in_image, li])

    # Downsample
    fe = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(merge)
    fe = LeakyReLU(alpha=alpha)(fe)
    fe = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(fe)
    fe = LeakyReLU(alpha=alpha)(fe)
    fe = Flatten()(fe)
    fe = Dropout(dropout_rate)(fe)
    out_layer = Dense(1, activation='sigmoid')(fe)

    model = Model([in_image, in_label], out_layer)
    opt = Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

def define_generator(latent_dim, n_classes=10, alpha=0.2):
    # Label input
    in_label = Input(shape=(1,))
    li = Embedding(n_classes, 50)(in_label)
    n_nodes = 8 * 8
    li = Dense(n_nodes)(li)
    li = Reshape((8, 8, 1))(li)

    # Image generator input
    in_lat = Input(shape=(latent_dim,))

    # Foundation for 8x8 image
    n_nodes = 128 * 8 * 8
    gen = Dense(n_nodes)(in_lat)
    gen = LeakyReLU(alpha=alpha)(gen)
    gen = Reshape((8, 8, 128))(gen)

    # Merge image gen and label input
    merge = Concatenate()([gen, li])

    # Upsample to 16x16
    gen = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(merge)
    gen = LeakyReLU(alpha=alpha)(gen)
    # Upsample to 32x32
    gen = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(gen)
    gen = LeakyReLU(alpha=alpha)(gen)

    # Output
    out_layer = Conv2D(3, (8, 8), activation='sigmoid', padding='same')(gen)

    model = Model([in_lat, in_label], out_layer)
    return model

def define_gan(g_model, d_model, learning_rate=0.0002, beta_1=0.5, beta_2=0.999):
    d_model.trainable = False
    # Connect generator and discriminator
    gen_noise, gen_label = g_model.input
    gen_output = g_model.output
    gan_output = d_model([gen_output, gen_label])
    # Define gan model as taking noise and label and outputting a classification
    model = Model([gen_noise, gen_label], gan_output)
    # Compile model
    opt = Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model

def generate_real_samples(dataset, n_samples):
    iterator = iter(dataset)
    real_batch = next(iterator)
    X, labels = real_batch[0], real_batch[1]

    # Ensure the number of requested samples is not greater than the batch size
    n_samples = tf.minimum(n_samples, tf.shape(X)[0])

    # Select a subset of samples
    X = X[:n_samples]
    labels = labels[:n_samples]

    # Generate labels for the discriminator (all set to 1 for real samples)
    y = tf.ones((n_samples, 1), dtype=tf.float32)

    return [X, labels], y

def generate_latent_points(latent_dim, n_samples, n_classes):
    x_input = np.random.randn(latent_dim * n_samples)
    z_input = x_input.reshape(n_samples, latent_dim)
    labels = np.random.randint(0, n_classes, n_samples)
    return [z_input, labels]

def generate_fake_samples(generator, latent_dim, n_samples, n_classes):
    # Generate random latent points and labels using TensorFlow
    z_input, labels_input = generate_latent_points(latent_dim, n_samples, n_classes)

    # Convert latent points and labels to TensorFlow tensors
    z_input = tf.constant(z_input, dtype=tf.float32)
    labels_input = tf.constant(labels_input, dtype=tf.int64)

    # Generate fake images using the generator model
    images = generator([z_input, labels_input])

    # Generate labels for the discriminator (all set to 0 for fake samples)
    y = tf.zeros((n_samples, 1), dtype=tf.float32)

    return [images, labels_input], y

def train_and_evaluate(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=128, num_classes=10, callback=None):
    evaluator = ModelEvaluator(batch_size=args.eval_batch_size)
    
    bat_per_epo = args.buffer_size // args.batch_size
    half_batch = n_batch // 2

    for i in range(n_epochs):
        d_losses = []
        g_losses = []
        total_losses = []

        for j in range(bat_per_epo):
            # Train the discriminator on real and fake images separately (half batch each)
            [X_real, labels_real], y_real = generate_real_samples(dataset, half_batch)
            d_loss_real, _ = d_model.train_on_batch([X_real, labels_real], y_real)

            [X_fake, labels], y_fake = generate_fake_samples(g_model, latent_dim, half_batch, num_classes)
            d_loss_fake, _ = d_model.train_on_batch([X_fake, labels], y_fake)
            
            # Calculate total discriminator loss
            d_loss = d_loss_real + d_loss_fake
            d_losses.append(d_loss)

            # prepare points in latent space as input for the generator
            [z_input, labels_input] = generate_latent_points(latent_dim, n_batch, num_classes)

            # The generator wants the discriminator to label the generated samples as valid (ones)
            y_gan = np.ones((n_batch, 1))
            g_loss = gan_model.train_on_batch([z_input, labels_input], y_gan)
            g_losses.append(g_loss)
            
            total_loss = d_loss + g_loss
            total_losses.append(total_loss)

        # Calculate mean losses for the epoch
        mean_d_loss = np.mean(d_losses)
        mean_g_loss = np.mean(g_losses)
        mean_total_loss = np.mean(total_losses)

        if callback is not None:
            callback.on_epoch_end(i + 1)
            
        # Print losses on this epoch
        print('Epoch=%d, D_loss=%.3f, G_loss=%.3f, Total_loss=%.3f' % (i + 1, mean_d_loss, mean_g_loss, mean_total_loss))

        # Evaluate and log at specified epochs
        if (i + 1) % args.eval_freq == 0:
            number_of_samples = max(args.fid_gen_samples, args.inception_score_samples, args.wasserstein_distance_samples)
            
            real_images_for_evaluation, _ = generate_real_samples(dataset, n_samples=number_of_samples)
            [z_input_eval, labels_input_eval] = generate_latent_points(latent_dim, n_samples=number_of_samples, n_classes=num_classes)
            generated_images_for_evaluation = g_model.predict([z_input_eval, labels_input_eval])

            real_images_array = real_images_for_evaluation[0]
            gen_images_array = generated_images_for_evaluation
            gen_images_array = np.clip(gen_images_array, 0.0, 1.0)
            
            is_avg, is_std = evaluator.calculate_inception_score(gen_images_array[:args.inception_score_samples])
            wasserstein_distance = evaluator.calculate_wasserstein_distance(
                real_images_array[:args.wasserstein_distance_samples],
                gen_images_array[:args.wasserstein_distance_samples]
            )
            fid_score = evaluator.calculate_fid(
                real_images_array[:args.fid_real_samples],
                gen_images_array[:args.fid_gen_samples]
            )
            print(
                f'Epoch= {i + 1}, FID= {fid_score}, Inception Score= {is_avg:.4f} ± {is_std:.4f}, Wasserstein Distance= {wasserstein_distance}')
            
            mlflow.log_metric("FID Score", fid_score, step=i + 1)
            mlflow.log_metric("Avg. Inceprion Score", is_avg, step=i + 1)
            mlflow.log_metric("Std. Inception Score", is_std, step=i + 1)
            mlflow.log_metric("Wasserstein Distance", wasserstein_distance, step=i + 1)
            mlflow.log_metric("Epoch", i+1, step=i + 1)
            
        mlflow.log_metric("Desc. Loss", mean_d_loss, step=i + 1)
        mlflow.log_metric("Gen. Loss", mean_g_loss, step=i + 1)
        mlflow.log_metric("Total Loss", mean_total_loss, step=i + 1)
        
    # Save model weights at the end of training
    g_model.save_weights(os.path.join(save_callback.log_folder, 'final_generator_weights.h5'))
     
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Conditional GAN for Different Datasets')
    parser.add_argument('--latent_dim', type=int, default=100, help='Size of the latent space')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--buffer_size', type=int, default=50000, help='Buffer size for shuffling')
    parser.add_argument('--dataset', type=str, default='fashion_mnist', choices=['fashion_mnist', 'cifar10', 'svhn' ,'imagenet'], help='Dataset name')
    parser.add_argument('--learning_rate', type=float, default=0.0002, help='Learning rate for Adam optimizer')
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha value for LeakyReLU')
    parser.add_argument('--beta_1', type=float, default=0.5, help='Beta_1 value for Adam optimizer')
    parser.add_argument('--beta_2', type=float, default=0.999, help='Beta_2 value for Adam optimizer')
    parser.add_argument('--dropout_rate', type=float, default=0.4, help='Dropout rate in the discriminator')
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

    latent_dim = args.latent_dim
        
    # Initialize MLflow and create an experiment
    # os.environ['MLFLOW_TRACKING_URI'] = 'file:///tmp/mlflow'
    mlflow.set_experiment(f'CGAN_{args.dataset}_exp_{args.exp_no}')
    mlflow.start_run()
    mlflow.set_tags({"model": "CGAN", "dataset": args.dataset, "exp_no": args.exp_no})
    
    dataset, num_classes = load_dataset(args.dataset, buffer_size=args.buffer_size, batch_size=args.batch_size)
    
    d_model = define_discriminator(n_classes=num_classes, dropout_rate=args.dropout_rate, learning_rate=args.learning_rate, alpha=args.alpha, beta_1=args.beta_1, beta_2=args.beta_2)
    g_model = define_generator(latent_dim, alpha=args.alpha, n_classes=num_classes)
    gan_model = define_gan(g_model, d_model, learning_rate=args.learning_rate, beta_1=args.beta_1, beta_2=args.beta_2)

    save_callback = SaveCallback(
        model_name='CGAN',
        dataset_name=args.dataset,
        num_classes=num_classes,
        generator=g_model,
        latent_dim=latent_dim,
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
        
    train_and_evaluate(
        g_model=g_model,
        d_model=d_model,
        gan_model= gan_model,
        dataset= dataset,
        latent_dim= latent_dim,
        n_epochs=args.epochs,
        n_batch=args.batch_size,
        num_classes=num_classes,
        callback=save_callback
    )
    
    # # Log model artifacts
    # mlflow.log_artifacts(save_callback.log_folder, artifact_path="logs")

    # End the MLflow run
    mlflow.end_run()