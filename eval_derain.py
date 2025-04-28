import os, sys, math, argparse, json
import numpy as np
from tqdm.auto import tqdm
from skimage import img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity   as ssim_loss

import tensorflow as tf
from models.ast_tf import ASTModel

tf.keras.mixed_precision.set_global_policy('float32')  
tf.get_logger().setLevel('ERROR')    

def mkdir(path):
    os.makedirs(path, exist_ok=True)
    
def split_image(tensor, crop_size=1152, overlap_size=384):
    """
    tensor: 4-D tf.Tensor  [1, H, W, 3]  (float32 0-1)
    returns: list of crops (tf.Tensor), list of (top,left) starts
    """
    _, H, W, _ = tensor.shape
    stride = crop_size - overlap_size
    ys = list(range(0, max(H - crop_size, 0) + 1, stride))
    xs = list(range(0, max(W - crop_size, 0) + 1, stride))
    crops, starts = [], []
    for y in ys:
        for x in xs:
            crop = tensor[:, y:y+crop_size, x:x+crop_size, :]
            crop = tf.image.resize_with_crop_or_pad(crop, crop_size, crop_size)
            crops.append(crop)
            starts.append((y, x))
    return crops, starts, (H, W)

def merge_image(crops, starts, orig_hw, crop_size=1152, overlap_size=384):
    """inverse of split_image, puts crops back together with simple averaging"""
    H, W = orig_hw
    acc = tf.zeros([H, W, 3], tf.float32)
    count = tf.zeros([H, W, 3], tf.float32)

    for crop, (y, x) in zip(crops, starts):
        crop = crop[0]
        crop_shape = tf.shape(crop)

        indices = tf.stack(tf.meshgrid(
            tf.range(y, y + crop_shape[0]),
            tf.range(x, x + crop_shape[1]),
            indexing='ij'
        ), axis=-1)

        indices = tf.reshape(indices, [-1, 2])

        updates = tf.reshape(crop, [-1, 3])

        acc = tf.tensor_scatter_nd_add(acc, indices, updates)
        count = tf.tensor_scatter_nd_add(count, indices, tf.ones_like(updates))

    return acc / tf.maximum(count, 1e-6)

parser = argparse.ArgumentParser(description='AST-TF Dense-Haze evaluation')
parser.add_argument('--input_dir',  required=True,  help='Dense-Haze root with GT/INPUT PNGs')
parser.add_argument('--result_dir', default='./results_densehaze_tf', help='Folder to save outputs')
parser.add_argument('--weights',    required=True,  help='.h5 or SavedModel dir')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--save_images', action='store_true')
parser.add_argument('--crop_size',     type=int, default=64)
parser.add_argument('--overlap_size',  type=int, default=16)
args = parser.parse_args()

mkdir(args.result_dir)

# Load saved model
ckpt_dir = args.weights    

model = ASTModel(embed_dim=16)
_ = model(tf.zeros([1, 64, 64, 3]))   
ckpt = tf.train.Checkpoint(model=model)

print("Restoring", ckpt_dir)
ckpt.restore(ckpt_dir).expect_partial()  

# Load validation dataset
from data.dataset_loader import get_dataset

_, _, test_ds = get_dataset(
    input_dir = os.path.join(args.input_dir, 'input'),
    gt_dir = os.path.join(args.input_dir, 'gt'),
    patch_size = 0,
    batch_size = args.batch_size,
)

def attach_filename(inp, gt):
    fname = tf.strings.reduce_join(
        tf.strings.split(tf.data.experimental.get_single_element(inp.filename))[-1])
    return inp, gt, fname

psnr_vals, ssim_vals = [], []

for inp, gt in tqdm(test_ds, desc='Evaluating', unit='img'):
    img_tensor = inp['image']
    H, W = gt.shape[1], gt.shape[2]
    crops, starts, hw = split_image(img_tensor, args.crop_size, args.overlap_size)
    out_crops = [model(c, training=False) for c in crops]
    pred = merge_image(out_crops, starts, hw, args.crop_size, args.overlap_size)
    pred = tf.clip_by_value(pred, 0., 1.).numpy()

    gt_img = gt[0].numpy()
    psnr = psnr_loss(pred, gt_img)
    ssim = ssim_loss(pred, gt_img, channel_axis=2, data_range=1.)

    psnr_vals.append(psnr)
    ssim_vals.append(ssim)

    # save
    if args.save_images:
        fname = f"img_{len(psnr_vals):04d}.png"
        tf.keras.utils.save_img(os.path.join(args.result_dir, fname), img_as_ubyte(pred))

avg_psnr = float(np.mean(psnr_vals))
avg_ssim = float(np.mean(ssim_vals))
print(f"\nFinished Avg PSNR: {avg_psnr:.3f}  |  Avg SSIM: {avg_ssim:.4f}")

with open(os.path.join(args.result_dir, 'psnr_ssim.txt'), 'w') as f:
    for i, (p,s) in enumerate(zip(psnr_vals, ssim_vals)):
        f.write(f"Image {i:04d}: PSNR {p:.4f}  SSIM {s:.4f}\n")
    f.write(f"\nAVERAGE  PSNR {avg_psnr:.4f}  SSIM {avg_ssim:.4f}\n") 