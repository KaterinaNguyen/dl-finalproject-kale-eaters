# Deep Learning Final Project

## Contributors

Emma Chen, Leanne Chia, Albert Dong, Katerina Nguyen

## Setup

### 1. Clone the Repository

Run:

```
git clone https://github.com/KaterinaNguyen/dl-finalproject-kale-eaters.git
```

### 2. Virtual Environment

Run:

```zsh
cd dl-finalproject-kale-eaters
```

Make sure you have Python 3.10 installed (e.g. via Homebrew: brew install python@3.10). Then run:

```zsh
/opt/homebrew/bin/python3.10 -m venv tf-venv
source tf-venv/bin/activate
```

### 3. Install Requirements

Inside the virtual environment, run:

```zsh
pip install --upgrade pip
```

Then, if you're on Mac:

```zsh
pip install tensorflow-macos
pip install -r requirements.txt
```

Otherwise:

```zsh
pip install tensorflow
pip install -r requirements.txt
```

### 4. Data

- Download the data here: https://drive.google.com/file/d/1cMXWICiblTsRl1zjN8FizF5hXOpVOJz4/view
- Unzip the downloaded folder
- Navigate to `DID-MDN-datasets/DID-MDN-training/Rain_Medium`
- Move the images from this folder into your git repository with: `mv path/to/source/*.jpg DID-MDN-all`
- From your terminal, run: `python3 split_pairs_in_dataset.py`
- Navigate to `DID-MDN-split/gt` to access the 'ground truth' images
- Navigate to `DID-MDN-split/input` tp access the synthetically altered rain images

### 5. Evaluation

Run:

```zsh
python testing_derain.py --input_dir DID-MDN-split --weights train_v2_checkpoints_ast/ckpt-10 --crop 64 --overlap 16 --batch_crops 8 --save_images
```

## Relevant Files

- `CSCI 1470 Final Writeup_Reflection.pdf`
- `poster.jpg`
- `/data/dataset_loader.py`: data preprocessing
- `split_pairs_in_dataset.py`: splitting ground truth and input images
- `/models`: model components
- `train_derain.py`: model training
- `testing_derain.py`: testing and evaluation
- `/eval_results`: model outputs from testing
