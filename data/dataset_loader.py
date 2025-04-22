import os
import tensorflow as tf
import numpy as np
from PIL import Image
import random

def load_image(path):
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.float32) / 255.0

def get_dataset(input_dir, gt_dir, patch_size=256, batch_size=4, split=(0.8, 0.1, 0.1), seed=42):
    input_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(".png")])
    gt_files = sorted([f for f in os.listdir(gt_dir) if f.lower().endswith(".png")])

    # Match files based on name
    common_files = sorted(list(set(input_files).intersection(set(gt_files))))
    input_paths = [os.path.join(input_dir, f) for f in common_files]
    gt_paths = [os.path.join(gt_dir, f) for f in common_files]

    # Shuffle and split
    combined = list(zip(input_paths, gt_paths))
    random.seed(seed)
    random.shuffle(combined)
    total = len(combined)
    n_train = int(split[0] * total)
    n_val = int(split[1] * total)

    train_pairs = combined[:n_train]
    val_pairs = combined[n_train:n_train + n_val]
    test_pairs = combined[n_train + n_val:]

    train_ds = make_dataset(train_pairs, patch_size, batch_size, is_training=True)
    val_ds = make_dataset(val_pairs, patch_size, batch_size, is_training=False)
    test_ds = make_dataset(test_pairs, patch_size, batch_size, is_training=False)

    return train_ds, val_ds, test_ds

def make_dataset(pairs, patch_size, batch_size, is_training):
    def generator():
        for inp_path, gt_path in pairs:
            inp = load_image(inp_path)
            gt = load_image(gt_path)
            yield inp, gt

    def preprocess(inp, gt):
        inp = tf.convert_to_tensor(inp, dtype=tf.float32)
        gt = tf.convert_to_tensor(gt, dtype=tf.float32)

        stacked = tf.stack([inp, gt], axis=0)

        if is_training:
            stacked = tf.image.random_crop(stacked, size=(2, patch_size, patch_size, 3))
            stacked = tf.image.random_flip_left_right(stacked)
            stacked = tf.image.random_flip_up_down(stacked)
        else:
            stacked = tf.image.resize_with_crop_or_pad(stacked, patch_size, patch_size)

        return stacked[0], stacked[1]

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32)
        )
    )

    dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset