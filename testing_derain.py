#!/usr/bin/env python3

import os, argparse, time, json
import numpy as np
from tqdm.auto import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage import img_as_ubyte

import tensorflow as tf
from models.ast_tf import ASTModel
from data.dataset_loader import get_dataset 

tf.keras.mixed_precision.set_global_policy("float32")
tf.get_logger().setLevel("ERROR")            


def mkdir(path: str):
    os.makedirs(path, exist_ok=True)

def split_image(img4d: tf.Tensor, crop, overlap):
    """
    Args
        img4d : [1,H,W,3]  float32 (0-1)
    Returns
        crops, starts, (H,W)
    """
    if crop == 0:                      
        h, w = img4d.shape[1:3]
        return [img4d], [(0, 0)], (h, w)

    stride  = max(1, crop - overlap)     
    _, H, W, _ = img4d.shape
    ys = list(range(0, max(H - crop, 0) + 1, stride))
    xs = list(range(0, max(W - crop, 0) + 1, stride))

    crops, starts = [], []
    for y in ys:
        for x in xs:
            patch = img4d[:, y:y+crop, x:x+crop, :]
            patch = tf.image.resize_with_crop_or_pad(patch, crop, crop)
            crops.append(patch)
            starts.append((y, x))
    return crops, starts, (H, W)

def batched_forward(model, crops, bs=8):
    """Run list[crop] through model in small batches to save VRAM."""
    outs = []
    for i in range(0, len(crops), bs):
        batch = tf.concat(crops[i:i+bs], 0)      
        outs.append(model(batch, training=False))  
    return tf.split(tf.concat(outs, 0), len(crops), axis=0)

@tf.function
def merge_patches(patches, starts, H, W):
    acc = tf.zeros([H, W, 3], tf.float32)
    count = tf.zeros([H, W, 3], tf.float32)

    for patch, (y, x) in zip(patches, starts):
        h = tf.shape(patch)[1]
        w = tf.shape(patch)[2]

        idx_y, idx_x = tf.meshgrid(tf.range(y, y+h), tf.range(x, x+w), indexing='ij')
        indices = tf.stack([tf.reshape(idx_y, [-1]),
                            tf.reshape(idx_x, [-1])], axis=1)

        updates = tf.reshape(patch[0], [-1, 3])
        acc = tf.tensor_scatter_nd_add(acc, indices, updates)
        count = tf.tensor_scatter_nd_add(count, indices, tf.ones_like(updates))

    return acc / tf.maximum(count, 1e-6)

def main(cfg):
    mkdir(cfg.result_dir)

    _, _, test_ds = get_dataset(
        input_dir = os.path.join(cfg.input_dir, "input"),
        gt_dir = os.path.join(cfg.input_dir, "gt"),
        patch_size=0,                  
        batch_size=1,                   
    )

    model = ASTModel(embed_dim=16)
    _ = model(tf.zeros([1, 64, 64, 3]))    

    ckpt = tf.train.Checkpoint(model=model)
    print("Restoring", cfg.weights)
    ckpt.restore(cfg.weights).expect_partial()

    psnr_all, ssim_all = [], []
    t0 = time.time()

    for idx, (features, gt) in enumerate(tqdm(test_ds, desc="Evaluating", unit="img")):
        img = features["image"]            
        crops, starts, (H, W) = split_image(img, cfg.crop, cfg.overlap)

        pred_patches = batched_forward(model, crops, bs=cfg.batch_crops)
        pred = merge_patches(pred_patches, starts, H, W)
        pred = tf.clip_by_value(pred, 0., 1.).numpy()

        gt_img = gt[0].numpy()
        psnr = peak_signal_noise_ratio(gt_img, pred, data_range=1.)
        ssim = structural_similarity(gt_img, pred, channel_axis=2, data_range=1.)

        psnr_all.append(psnr)
        ssim_all.append(ssim)

        if cfg.save_images:
            out_name = f"img_{idx:04d}_{psnr:.2f}dB.png"
            tf.keras.utils.save_img(os.path.join(cfg.result_dir, out_name),
                                    img_as_ubyte(pred))

    print(f"\nFinished in {time.time()-t0:.1f}s  |"
          f"  avg-PSNR {np.mean(psnr_all):.3f} dB"
          f"  avg-SSIM {np.mean(ssim_all):.4f}")

    with open(os.path.join(cfg.result_dir, "metrics.json"), "w") as f:
        json.dump({"PSNR": float(np.mean(psnr_all)),
                   "SSIM": float(np.mean(ssim_all))}, f, indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser("AST TensorFlow evaluation")
    p.add_argument("--input_dir",  required=True,  help="dataset root with input/ gt/")
    p.add_argument("--weights",    required=True,  help="checkpoint path e.g. train_v2_checkpoints_ast/ckpt-10")
    p.add_argument("--result_dir", default="eval_results", help="folder to save outputs")
    p.add_argument("--crop",       type=int, default=64,  help="crop size (0 = full frame)")
    p.add_argument("--overlap",    type=int, default=16,  help="overlap between crops")
    p.add_argument("--batch_crops",type=int, default=8,   help="how many patches per forward pass")
    p.add_argument("--save_images",action="store_true")
    cfg = p.parse_args()

    if cfg.overlap >= cfg.crop and cfg.crop > 0:
        raise ValueError("--overlap must be < --crop")

    main(cfg)