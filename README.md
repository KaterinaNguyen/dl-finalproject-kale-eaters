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
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Requirements
Inside the virtual environment, run:
```zsh
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


