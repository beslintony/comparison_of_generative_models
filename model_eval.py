import numpy as np
import tensorflow as tf
from scipy.linalg import sqrtm
import ot 

class ModelEvaluator:
    def __init__(self, batch_size=64):
        self.batch_size = batch_size
        self.inception_model = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False, pooling='avg')

    def _inception_activations(self, images):
        size = 299
        images = tf.image.resize(images, [size, size], method=tf.image.ResizeMethod.BILINEAR)
        activations = self.inception_model(images)
        return activations

    def _get_inception_activations(self, inps):
        act = np.zeros([inps.shape[0], 2048], dtype=np.float32)

        for i in range(0, inps.shape[0], self.batch_size):
            inp = inps[i: i + self.batch_size]
            # inp = tf.keras.applications.inception_v3.preprocess_input(inp)
            act[i: i + self.batch_size] = self._inception_activations(inp).numpy()

        return act

    def _calculate_fid(self, activations1, activations2):
        mu1, sigma1 = np.mean(activations1, axis=0), np.cov(activations1, rowvar=False)
        mu2, sigma2 = np.mean(activations2, axis=0), np.cov(activations2, rowvar=False)

        diff = mu1 - mu2
        covmean = sqrtm(sigma1.dot(sigma2))

        # Calculate FID
        fid = np.sum(diff**2) + np.trace(sigma1 + sigma2 - 2*covmean.real)

        return fid

    def calculate_fid(self, real_images, generated_images):
        act_real = self._get_inception_activations(real_images)
        act_generated = self._get_inception_activations(generated_images)
        fid = self._calculate_fid(act_real, act_generated)
        return fid

    def _calculate_kl_divergence(self, p_yx, p_y, eps=1E-16):
        kl_d = p_yx * (tf.math.log(p_yx + eps) - tf.math.log(p_y + eps))
        sum_kl_d = tf.reduce_sum(kl_d, axis=1)
        avg_kl_d = tf.reduce_mean(sum_kl_d)
        return avg_kl_d

    def _calculate_wasserstein_distance(self, images_set1, images_set2):
        if isinstance(images_set1, tf.Tensor):
            images_set1_np = images_set1.numpy()
        else:
            images_set1_np = images_set1
        
        if isinstance(images_set2, tf.Tensor):
            images_set2_np = images_set2.numpy()
        else:
            images_set2_np = images_set2
        
        flat_images_set1 = images_set1_np.reshape((images_set1_np.shape[0], -1))
        flat_images_set2 = images_set2_np.reshape((images_set2_np.shape[0], -1))

        if np.max(flat_images_set1) > 1:
            flat_images_set1 = flat_images_set1.astype(np.float32) / 255.0

        if np.max(flat_images_set2) > 1:
            flat_images_set2 = flat_images_set2.astype(np.float32) / 255.0

        M = ot.dist(flat_images_set1, flat_images_set2)
        a, b = np.ones((images_set1_np.shape[0],)) / images_set1_np.shape[0], np.ones((images_set2_np.shape[0],)) / images_set2_np.shape[0]
        wasserstein_distance = ot.emd2(a, b, M)
        return wasserstein_distance

    def calculate_inception_score(self, images, n_split=10):
        model = tf.keras.applications.InceptionV3()
        images = images.astype('float32')
        # images = tf.keras.applications.inception_v3.preprocess_input(images)
        scores = []
        n_part = images.shape[0] // n_split
        for i in range(n_split):
            dataset = tf.data.Dataset.from_tensor_slices(images[i * n_part: (i + 1) * n_part])
            dataset = dataset.batch(self.batch_size)
            p_yx_split = []
            for batch in dataset:
                resized_batch = tf.image.resize(batch, (299, 299))
                p_yx_batch = model.predict(resized_batch)
                p_yx_split.append(p_yx_batch)
            p_yx_split = tf.concat(p_yx_split, axis=0)
            p_y_split = tf.reduce_mean(p_yx_split, axis=0, keepdims=True)
            avg_kl_d_split = self._calculate_kl_divergence(p_yx_split, p_y_split)
            is_score = tf.exp(avg_kl_d_split).numpy()
            scores.append(is_score)
        is_avg, is_std = np.mean(scores), np.std(scores)
        return is_avg, is_std

    def calculate_wasserstein_distance(self, real_images, generated_images):
        wasserstein_distance = self._calculate_wasserstein_distance(real_images, generated_images)
        return wasserstein_distance
