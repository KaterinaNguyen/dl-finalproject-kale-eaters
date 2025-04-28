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

### Evaluatoin
- Run: 

### UPDATES:
- Verified that dataset_loader.py seems to work by checking that batch shapes print out when I run:
```zsh
python3 test_dataset_loader.py
```
test_dataset_loader.py calls on functions in dataset_loader.py

- ASSA and FRFN implemented in assa.py and frfn.py

- So we now have working TensorFlow reimplementations of:
    1. ASSA in assa.py
    2. FRFN in frfn.py
    3. AST Model – Full Encoder-Bottleneck-Decoder transformer architecture in ast_tf.py

    Each file mirrors the logic from the original PyTorch code, but translated and adapted for TensorFlow/Keras. After this, we will need to use these components for:
    - Training in some python file (**author's code has numerous train files in their train folder that we could maybe reference? but it's written with pytorch -- this is step 8 in our process**)
        - Note: this training file is also the file where we need to write functions to import data and actually run training on the data
    
- test_ast_tf.py checks that the pipeline does work fine (doesn't train, so it won't give a good model output, but it does show that the model doesn't have NAN values)

- try_train.py is an attempt at doing a small training for the model and it already takes super long to run. didn't manage to finish running it becuase the dataset is huge and each epoch literally has 3200 steps because of that, so not sure how performance is -- will need more people to work on this more.
    - Seems prospective though! I trained 3 epochs on the full dataset and by epoch 3 there's at least like some colored pixels whereas epoch 1 and 2 were completely black screen -- I think will need to **run training overnight for sure**. It took 2 hours to run 3 epochs... crying rn. 
    - But yes please don't use this python file as the actual training file... it was pretty carelessly written just for my reference/gauge! Write the actual training file based on how the writers did it. Oh, I also wrote charbonnier loss in losses.py (didn't use it from losses.py in try_train.py but ideally in the actual model we should use the charbonnier loss as implemented in losses.py)

**Note**: Theoretically test_dataset_loader.py, test_ast_tf.py, and try_train.py are all just files to test implementation, NOT real files part of our final submission with model and everything! Still need to write training files and evaluation files etc.!