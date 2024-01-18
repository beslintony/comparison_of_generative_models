import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import os
from tqdm import tqdm

# Constants
IMSIZE = 32
BATCH_SIZE = 32

# Paths to your validation dataset
val_data_dir = 'imagenet/val'
mapping_file_path = 'imagenet/imagenet/LOC_synset_mapping.txt'

# Function to load and preprocess images
def load_and_preprocess_image(image_path, label):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3)
    
    # Check if the image is grayscale and convert it to color
    if img.shape[-1] == 1:
        img = tf.image.grayscale_to_rgb(img)
    
    img = tf.image.resize(img, [IMSIZE, IMSIZE])
    img = img / 255.0  # Normalize pixel values
    return img, label

# Create a list of image paths and corresponding labels
def load_image_paths_and_labels(data_dir):
    image_paths = []
    labels = []

    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            image_paths.extend(tf.data.Dataset.list_files(os.path.join(class_path, '*')))
            labels.extend([class_name] * len(os.listdir(class_path)))

    return image_paths, labels

# Load image paths and labels for validation
val_image_paths, val_labels = load_image_paths_and_labels(val_data_dir)

# Load the label mapping from the LOC_synset_mapping.txt file
class_mapping = {}
with open(mapping_file_path, 'r') as file:
    for line in file:
        parts = line.strip().split(' ')
        class_id = parts[0]
        class_name = ' '.join(parts[1:])
        class_mapping[class_id] = class_name

# Convert class names to numerical labels using the mapping
val_labels = [class_mapping[label] for label in val_labels]

# Create a mapping from class names to integer labels
class_to_label = {class_name: idx for idx, class_name in enumerate(set(val_labels))}
num_classes = len(class_to_label)

# Convert class names to numerical labels
val_labels = [class_to_label[label] for label in val_labels]

# Print the content of val_labels
print("val_labels:", val_labels)

# Convert numerical labels to one-hot encoded labels
val_labels_one_hot = to_categorical(val_labels, num_classes=num_classes)

# Function to serialize the sample
def serialize_sample(image, label):
    image = tf.image.convert_image_dtype(image, dtype=tf.uint8)  # Convert to uint8

    feature = {
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(image).numpy()])),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
    }
    sample_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return sample_proto.SerializeToString()

# Write TFRecord file for validation
val_tfrecord_filename = 'val_data.tfrecord'
with tf.io.TFRecordWriter(val_tfrecord_filename) as writer:
    for image_path, label in tqdm(zip(val_image_paths, val_labels), total=len(val_image_paths), desc="Creating TFRecord"):
        img, lbl = load_and_preprocess_image(image_path, label)
        tf_sample = serialize_sample(img, lbl)
        writer.write(tf_sample)

print(f"TFRecord for validation set created: {val_tfrecord_filename}")
