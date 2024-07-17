# UniRec: A Dual Enhancement of Uniformity and Frequency in Sequential Recommendations

This repository provides the code of experiments from the paper, which has been accepted by 33rd ACM International Conference on
Information and Knowledge Management (CIKM 2024). The paper is available online on [arXiv](https://arxiv.org/abs/2406.18470).

Our code structure is based on the [mojito](https://github.com/deezer/sigir23-mojito).

## Environment Requirement

The code runs well under python 3.9.13. The required packages are as follows:

- Tensorflow-gpu==2.11.0
- numpy==1.24.2
- scipy==1.10.1
- pandas==1.5.3
- keras==2.11.0
- tqdm==4.65.0
- toolz==0.12.0
- tqdm==4.65.0

## Datasets

You need to download the following datasets:
- [ml-1m](https://grouplens.org/datasets/movielens/1m/)
- [Amazon Beauty](https://jmcauley.ucsd.edu/data/amazon/)
- [Amazon Books](https://jmcauley.ucsd.edu/data/amazon/)
- [Amazon Toys](https://jmcauley.ucsd.edu/data/amazon/)

Place the datasets in the following directory structure:
```
exp/data/
├── ml1m
├── beauty
├── books
└── toy
```

## Data Preprocessing

Before running the model, you need to preprocess the data. The preprocessing scripts are located in `exp/data/ml1m`.


## Running the Model

To quickly run the UniRec model, use the provided bash script:
```bash
bash run_unirec.sh
```

This script will execute the training and evaluation processes of the UniRec model.


## Hyperparameters
All hyperparameters are saved in the `/configs` directory.
