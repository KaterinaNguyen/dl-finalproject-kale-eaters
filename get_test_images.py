from data.dataset_loader import get_dataset
import tensorflow as tf 
import os

def mkdir(path):
    os.makedirs(path, exist_ok=True)

input_dir = "DID-MDN-split/input"
gt_dir = "DID-MDN-split/gt"
output_dir = "dumped_test_images"
mkdir(output_dir)

_, _, test_ds = get_dataset(
    input_dir = input_dir,
    gt_dir = gt_dir,
    patch_size = 0,
    batch_size = 1,
)

for inp, gt in test_ds:
    filename_tensor = inp['filename'][0]
    filename = os.path.basename(filename_tensor.numpy().decode('utf-8'))

    img    = inp['image'][0] 
    gt_img = gt[0]  

    tf.keras.utils.save_img(
        os.path.join(output_dir, f"input_{filename}"),
        img
    )
    tf.keras.utils.save_img(
        os.path.join(output_dir, f"gt_{filename}"),
        gt_img
    )

print(f"Dumped test set images to '{output_dir}'")