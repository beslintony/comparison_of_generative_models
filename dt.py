import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

def parse_imagenet_tfrecord_fn(sample_proto):
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    sample = tf.io.parse_single_example(sample_proto, feature_description)
    image = tf.io.decode_jpeg(sample['image'], channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    label = sample['label']
    return image, label

def preprocess_image(image, label, target_size=(32, 32)):
    image = tf.image.resize(image, size=target_size)
    image = tf.image.grayscale_to_rgb(image)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

def load_dataset(dataset_name, target_size=(32, 32), buffer_size=60000, batch_size=128, with_labels=True):
    num_classes = None

    if dataset_name == 'imagenet':
        num_classes = 1000
        imagenet_filename = '/home/ml-063/ml-new/repo/thesis/imagenet/train_data.tfrecord'
        tfrecord_dataset = tf.data.TFRecordDataset([imagenet_filename])
        parsed_dataset = tfrecord_dataset.map(parse_imagenet_tfrecord_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    elif dataset_name == 'fashion_mnist':
        ds, info = tfds.load(dataset_name, split='train', with_info=True)
        num_classes = info.features['label'].num_classes
        parsed_dataset = tf.data.Dataset.from_tensor_slices({
            'image': [sample['image'] for sample in tfds.as_numpy(ds)],
            'label': [sample['label'] for sample in tfds.as_numpy(ds)],
        })
        parsed_dataset = parsed_dataset.map(lambda x: preprocess_image(x['image'], x['label'], target_size), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    elif dataset_name == 'cifar10':
        ds, info = tfds.load(dataset_name, split='train', with_info=True)
        num_classes = info.features['label'].num_classes
        parsed_dataset = tf.data.Dataset.from_tensor_slices({
            'image': [sample['image'] for sample in tfds.as_numpy(ds)],
            'label': [sample['label'] for sample in tfds.as_numpy(ds)],
        })
        parsed_dataset = parsed_dataset.map(lambda x: (tf.cast(x['image'], tf.float32) / 255.0, x['label']), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    elif dataset_name == 'svhn':
        ds, info = tfds.load('svhn_cropped', split='train', with_info=True)
        num_classes = info.features['label'].num_classes
        parsed_dataset = tf.data.Dataset.from_tensor_slices({
            'image': [sample['image'] for sample in tfds.as_numpy(ds)],
            'label': [sample['label'] for sample in tfds.as_numpy(ds)],
        })
        parsed_dataset = parsed_dataset.map(lambda x: (tf.cast(x['image'], tf.float32) / 255.0, x['label']), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    parsed_dataset = parsed_dataset.shuffle(buffer_size=buffer_size).batch(batch_size, drop_remainder=True).repeat().prefetch(tf.data.experimental.AUTOTUNE)

    if with_labels:
        return parsed_dataset, num_classes
    else:
        return parsed_dataset.map(lambda x, y: x), num_classes

def load_validation_dataset(dataset_name, target_size=(32, 32), batch_size=128, with_labels=True):
    num_classes = None

    if dataset_name == 'imagenet':
        num_classes = 1000
        imagenet_val_filename = '/home/ml-063/ml-new/repo/thesis/imagenet/val_data.tfrecord'
        tfrecord_dataset = tf.data.TFRecordDataset([imagenet_val_filename])
        parsed_dataset = tfrecord_dataset.map(parse_imagenet_tfrecord_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    elif dataset_name == 'fashion_mnist':
        ds, info = tfds.load(dataset_name, split='test', with_info=True)
        num_classes = info.features['label'].num_classes
        parsed_dataset = tf.data.Dataset.from_tensor_slices({
            'image': [sample['image'] for sample in tfds.as_numpy(ds)],
            'label': [sample['label'] for sample in tfds.as_numpy(ds)],
        })
        parsed_dataset = parsed_dataset.map(lambda x: preprocess_image(x['image'], x['label'], target_size), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    elif dataset_name == 'cifar10':
        ds, info = tfds.load(dataset_name, split='test', with_info=True)
        num_classes = info.features['label'].num_classes
        parsed_dataset = tf.data.Dataset.from_tensor_slices({
            'image': [sample['image'] for sample in tfds.as_numpy(ds)],
            'label': [sample['label'] for sample in tfds.as_numpy(ds)],
        })
        parsed_dataset = parsed_dataset.map(lambda x: (tf.cast(x['image'], tf.float32) / 255.0, x['label']), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    elif dataset_name == 'svhn':
        ds, info = tfds.load('svhn_cropped', split='test', with_info=True)
        num_classes = info.features['label'].num_classes
        parsed_dataset = tf.data.Dataset.from_tensor_slices({
            'image': [sample['image'] for sample in tfds.as_numpy(ds)],
            'label': [sample['label'] for sample in tfds.as_numpy(ds)],
        })
        parsed_dataset = parsed_dataset.map(lambda x: (tf.cast(x['image'], tf.float32) / 255.0, x['label']), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    parsed_dataset = parsed_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    if with_labels:
        return parsed_dataset, num_classes
    else:
        return parsed_dataset.map(lambda x, y: x), num_classes
