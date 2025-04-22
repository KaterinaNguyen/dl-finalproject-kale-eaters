import tensorflow as tf
from data.dataset_loader import get_dataset
from models.ast_tf import ASTModel
import os

os.makedirs("outputs", exist_ok=True)

input_dir = "DID-MDN-split/input"
gt_dir = "DID-MDN-split/gt"
patch_size = 64
batch_size = 1
epochs = 100
learning_rate = 2e-4
log_freq = 10
save_model_path = "ast_model_trained.h5"

print("Loading dataset...")
train_ds, val_ds, _ = get_dataset(input_dir, gt_dir, patch_size=patch_size, batch_size=batch_size)

print("Building AST model...")
model = ASTModel()
_ = model(tf.keras.Input(shape=(patch_size, patch_size, 3)))
model.summary()

def charbonnier_loss(y_true, y_pred, epsilon=1e-3):
    diff = y_true - y_pred
    loss = tf.sqrt(tf.square(diff) + epsilon**2)
    return tf.reduce_mean(loss)

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

@tf.function
def train_step(degraded, clean):
    with tf.GradientTape() as tape:
        output = model(degraded, training=True)
        loss = charbonnier_loss(clean, output)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

@tf.function
def compute_psnr(clean, output):
    output = tf.clip_by_value(output, 0.0, 1.0)
    clean = tf.clip_by_value(clean, 0.0, 1.0)
    return tf.image.psnr(clean, output, max_val=1.0)

print("\nStarting training...")
for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    epoch_loss = 0.0
    for step, (degraded, clean) in enumerate(train_ds):
        loss = train_step(degraded, clean)
        epoch_loss += loss.numpy()

        if step % log_freq == 0:
            print(f"Step {step}, Loss: {loss.numpy():.6f}")

    avg_loss = epoch_loss / (step + 1)
    print(f"Epoch {epoch + 1} Loss: {avg_loss:.6f}")

    # Validation
    psnr_vals = []
    for degraded_val, clean_val in val_ds.take(5):
        output_val = model(degraded_val, training=False)
        psnr = compute_psnr(clean_val, output_val)
        psnr_vals.append(psnr.numpy())
    avg_psnr = sum(psnr_vals) / len(psnr_vals)
    print(f"Validation PSNR: {float(avg_psnr):.2f}")

    for degraded_val, clean_val in val_ds.take(1):
        output_val = model(degraded_val, training=False)
        output_val = tf.clip_by_value(output_val, 0.0, 1.0)

        # Save input, output, ground truth
        tf.keras.utils.save_img(f"outputs/epoch{epoch+1:02d}_input.png", degraded_val[0])
        tf.keras.utils.save_img(f"outputs/epoch{epoch+1:02d}_output.png", output_val[0])
        tf.keras.utils.save_img(f"outputs/epoch{epoch+1:02d}_gt.png", clean_val[0])
        break