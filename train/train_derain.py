import os, datetime, argparse, json
import tensorflow as tf

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.dataset_loader import get_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', type=str, default='DID-MDN-split/input')
parser.add_argument('--gt_dir', type=str, default='DID-MDN-split/gt')
parser.add_argument('--val_split', type=float, default=0.1)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--patch', type=int, default=64)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr_init', type=float, default=2e-4)
parser.add_argument('--warmup_ep', type=int, default=5)
parser.add_argument('--model_dir', type=str, default='checkpoints_ast')
args = parser.parse_args()

os.makedirs(args.model_dir, exist_ok=True)

# Enabling GPU
tf.config.set_visible_devices([], 'GPU')
# tf.keras.mixed_precision.set_global_policy('mixed_float16')
# tf.debugging.set_log_device_placement(True)

############## Data ##############
# from dataset_loader import get_dataset 

train_ds, val_ds, _ = get_dataset(
    input_dir=args.train_dir,
    gt_dir=args.gt_dir,
    patch_size=args.patch,
    batch_size=args.batch_size,
    split=(1.0 - args.val_split, args.val_split, 0.0)
)


############## Model, Loss, Optimizer ##############
from models.ast_tf import ASTModel
from losses import charbonnier_loss 

model = ASTModel()
_ = model(tf.random.uniform([1, args.patch, args.patch, 3]))

total_steps   = args.epochs * tf.data.experimental.cardinality(train_ds).numpy()
warmup_steps  = args.warmup_ep * tf.data.experimental.cardinality(train_ds).numpy()

def lr_schedule(step):
    step = tf.cast(step, tf.float32)
    lr_base = 0.5 * args.lr_init * (1 + tf.cos(3.14159265 *
                     (step - warmup_steps) / float(total_steps - warmup_steps)))
    lr_warm = args.lr_init * step / warmup_steps
    return tf.where(step < warmup_steps, lr_warm, lr_base)

# optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule,
#                                      beta_1=0.9, beta_2=0.999, epsilon=1e-8)
step_counter = tf.Variable(0, trainable=False)
optimizer = tf.keras.optimizers.Adam(learning_rate=lambda: lr_schedule(step_counter))

optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)


train_loss = tf.keras.metrics.Mean(name='train_loss')
val_psnr   = tf.keras.metrics.Mean(name='val_psnr')

ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, model=model)
manager = tf.train.CheckpointManager(ckpt, args.model_dir, max_to_keep=3)


@tf.function
def train_step(inp, gt):
    with tf.GradientTape() as tape:
        pred = model(inp, training=True)
        loss = charbonnier_loss(gt, pred)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
    scaled_grads = tape.gradient(loss, model.trainable_variables)
    grads = optimizer.get_unscaled_gradients(scaled_grads)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    train_loss(loss)

@tf.function
def val_step(inp, gt):
    pred = model(inp, training=False)
    psnr = tf.image.psnr(tf.clip_by_value(pred,0.,1.),
                         tf.clip_by_value(gt,   0.,1.), max_val=1.)
    val_psnr(psnr)

############## Training Loop ##############
for epoch in range(1, args.epochs + 1):
    train_loss.reset_state()
    val_psnr.reset_state()

    for x, y in train_ds:
        train_step(x, y)

    for x, y in val_ds:
        val_step(x, y)

    template = ("Epoch {:03d} | loss {:.5f} | val-psnr {:.2f} dB | lr {:.6e}")
    print(template.format(epoch,
                          train_loss.result(),
                          val_psnr.result(),
                          optimizer._optimizer.lr(ckpt.step)))

    # save checkpoint
    ckpt.step.assign_add(1)
    if epoch % 5 == 0:
        manager.save()

    # sample export
    if epoch % 10 == 0:
        sample_inp, _ = next(iter(val_ds))
        sample_out = model(sample_inp, training=False)
        tf.keras.utils.save_img(
            os.path.join(args.model_dir, f'sample_ep{epoch}.png'),
            tf.clip_by_value(sample_out[0], 0., 1.)
        )

print("Training complete. Best checkpoints are in", args.model_dir)