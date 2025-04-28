from data.dataset_loader import get_dataset

input_dir = "DID-MDN-split/input"
gt_dir = "DID-MDN-split/gt"

# Load datasets
train_ds, val_ds, test_ds = get_dataset(
    input_dir=input_dir,
    gt_dir=gt_dir,
    patch_size=256,
    batch_size=4
)

# Show one batch to confirm it works
for x, y in train_ds.take(1):
    print("Input batch shape:", x.shape)
    print("GT batch shape:", y.shape)

print("=== Train Batch ===")
for x, y in train_ds.take(1):
    print("Train input:", x.shape)
    print("Train GT:", y.shape)

print("=== Val Batch ===")
for x, y in val_ds.take(1):
    print("Val input:", x.shape)
    print("Val GT:", y.shape)

print("=== Test Batch ===")
for x, y in test_ds.take(1):
    print("Test input:", x.shape)
    print("Test GT:", y.shape)