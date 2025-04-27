import os, datetime, argparse, json
import tensorflow as tf
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.dataset_loader import get_dataset
import time
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
train_steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
val_steps_per_epoch   = tf.data.experimental.cardinality(val_ds).numpy()
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
        scaled_loss = optimizer.scale_loss(loss)

    scaled_grads = tape.gradient(scaled_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(scaled_grads, model.trainable_variables))
    train_loss.update_state(loss)
    step_counter.assign_add(1)
@tf.function
def val_step(inp, gt):
    pred = model(inp, training=False)
    psnr = tf.image.psnr(tf.clip_by_value(pred,0.,1.),
                         tf.clip_by_value(gt,   0.,1.), max_val=1.)
    val_psnr(psnr)

# Training loop
for epoch in range(1, args.epochs + 1):
    print(f"\n=== Epoch {epoch:03d}/{args.epochs} ===")
    epoch_start = time.time()
    
    train_loss.reset_state()
    val_psnr.reset_state()

    t = tqdm(train_ds, total=train_steps_per_epoch, desc="  train", unit="step")
    step_times = []
    
    for step, (x, y) in enumerate(t, start=1):
        step_start = time.time()
        train_step(x, y)
        step_times.append(time.time() - step_start)
        
        t.set_postfix({
            'step': step,
            'loss': f"{train_loss.result():.4f}",
            't/step': f"{(sum(step_times)/len(step_times)):.3f}s"
        })

    v = tqdm(val_ds, total=val_steps_per_epoch, desc="  valid", unit="step")
    for x, y in val_ds:
        val_step(x, y)

    epoch_time = time.time() - epoch_start
    avg_time_per_step = epoch_time / train_steps_per_epoch
    
    print((
        f"Epoch {epoch:03d} done in {epoch_time:.1f}s "
        f"({avg_time_per_step:.3f}s/step) | "
        f"loss={train_loss.result():.5f} | "
        f"val-psnr={val_psnr.result():.2f} dB"
    ))
    
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