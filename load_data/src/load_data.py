import os, pickle
import pandas as pd
import numpy as np
from pathlib import Path
import logging

def load_data(data_path, load_data_path):
    #pandas
    # logging.basicConfig(stream=sys.stdout)
    logger = logging.getLogger(__name__)
    #add log to std out

    train_data_path =  Path(data_path) / 'train.csv'
    test_data_path = Path(data_path) / 'test.csv'

    train_df = pd.read_csv(train_data_path)
    test_df = pd.read_csv(test_data_path)

    n_train = train_df.shape[0]
    n_test = test_df.shape[0]
    all_data = pd.concat([train_df, test_df]).reset_index(drop=True)

    logger.info(f"Train data shape: {train_df.shape}")

    os.makedirs(load_data_path, exist_ok=True)

    with open(f'{load_data_path}/all_data', 'wb') as f:
        pickle.dump((n_train, all_data), f)

    logger.info(f"Saved all_data.pickle to {load_data_path}")

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data-path', type=str, default='data')
parser.add_argument('--load-data-path', type=str, default='load_data')
args = parser.parse_args()

load_data(args.data_path, args.load_data_path)


