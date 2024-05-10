# README
# DLNLP_23-24_Final Assignment
This project aims at exploring NLP and deep learning techniques based on Hate/Offensive Speech Detection.

## Code Structure
* [`A`](./A/): Code for XHate implementation
    * [`Preprocessing.ipynb`](./A/Prepocessing.ipynb): acquire, analyse and preprocess raw data
    * [`Model.py`](./A/Model.py): define the architecture of classifiers
    * [`Dataset.py`](./A/Dataset.py): define dataset MyDataset
    * [`Train.py`](./A/Train.py): training script implemented with PyTorch
    * [`Test.py`](./A/Test.py): evaluation script
    * [`Fine_tuning.ipynb`](./A/Fine_tuning.ipynb): jupyter notebook for model fine-tuning

* [`Datasets`](./Datasets/): Twitter data for 
    * [`TweetHate.csv`](./Datasets/TweetHate.csv): raw data acquired by API
    * [`clean.csv`](./Datasets/clean.csv): preprocessed data

* [`main.py`](./main.py): Train the classifiers

## Installation and Requirements
The code requires common Python environments for model training and testing:
- Python 3.11.5
- PyTorch==1.3.1
- numpy==1.26.1
- pandas==2.1.1
- tqdm==4.66.1
- contractions
- symspellpy

