import tensorflow as tf
import time
from data.dataset_loader import get_dataset
from models.ast_tf import ASTModel
import os

input_dir = "DID-MDN-split/input"
gt_dir = "DID-MDN-split/gt"
patch_size = 128
batch_size = 1

print("Loading one batch from the dataset...")
train_ds, _, _ = get_dataset(input_dir, gt_dir, patch_size=patch_size, batch_size=batch_size)

print("Building AST model...")
model = ASTModel()
dummy_input = tf.keras.Input(shape=(patch_size, patch_size, 3))
model(dummy_input)
model.summary()

for degraded, clean in train_ds.take(1):
    print("Degraded input shape:", degraded.shape)
    print("Ground-truth shape: ", clean.shape)

    start = time.time()
    output = model(degraded)
    end = time.time()

    print("Output shape:", output.shape)
    print(f"Forward pass took {end - start:.4f} seconds")

    output = tf.clip_by_value(output, 0.0, 1.0)
    clean = tf.clip_by_value(clean, 0.0, 1.0)
    output = tf.cast(output, tf.float32)
    clean = tf.cast(clean, tf.float32)
    psnr = tf.image.psnr(clean, output, max_val=1.0)
    psnr = tf.where(tf.math.is_nan(psnr), tf.zeros_like(psnr), psnr)
    print("PSNR:", psnr.numpy())

import matplotlib.pyplot as plt

plt.subplot(1, 3, 1)
plt.imshow(degraded[0].numpy())
plt.title("Input (Degraded)")

plt.subplot(1, 3, 2)
plt.imshow(output[0].numpy())
plt.title("Model Output")

plt.subplot(1, 3, 3)
plt.imshow(clean[0].numpy())
plt.title("Ground Truth")

plt.show()
