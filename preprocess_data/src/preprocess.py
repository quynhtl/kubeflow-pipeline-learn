import os, joblib, pickle;
import json
from sklearn.model_selection import train_test_split
import logging
logger = logging.getLogger(__name__)

def preprocess_data(load_data_path: str, preprocess_data_path:str,
    test_size: float,
    mlpipeline_ui_metadata_path: str):
    #scikit-learn joblib, pandas


    with open(f'{load_data_path}/all_data', 'rb') as f:
        ntrain, all_data = pickle.load(f)

    # split features and label
    all_data_X = all_data.drop('label', axis=1)
    all_data_y = all_data.label

    # Reshape image in 3 dimensions (height = 28px, width = 28px , channel = 1)
    all_data_X = all_data_X.values.reshape(-1,28,28,1)

    # Normalize the data
    all_data_X = all_data_X / 255.0

    #Get the new dataset
    X = all_data_X[:ntrain].copy()
    y = all_data_y[:ntrain].copy()

    # split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    #creating the preprocess directory
    os.makedirs(preprocess_data_path, exist_ok = True)


    #Save the train_data as a pickle file to be used by the modelling component.
    with open(f'{preprocess_data_path}/train', 'wb') as f:
        pickle.dump((X_train,  y_train), f)

    #Save the test_data as a pickle file to be used by the predict component.
    with open(f'{preprocess_data_path}/test', 'wb') as f:
        pickle.dump((X_test,  y_test), f)


    print("Preprocessing done")
    print("Preprocessed data saved to {}".format(preprocess_data_path))
    print("Number of training samples: {}".format(len(X_train)))
    print("Number of test samples: {}".format(len(X_test)))

    os.makedirs(os.path.dirname(mlpipeline_ui_metadata_path), exist_ok=True)
    metadata = {
        'outputs' : [{
            'type': 'markdown',
            'storage': 'inline',
            'source': '''# Data Overview
**Number of training samples**: {}
**Number of test samples**: {}
'''.format(len(X_train), len(X_test))
        }]
    }
    with open(mlpipeline_ui_metadata_path, 'w') as f:
        json.dump(metadata, f)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--load-data-path", type=str, default="load_data")
parser.add_argument("--preprocess-data-path", type=str, default="preprocess_data_path")
parser.add_argument("--test-size", type=float, default=0.1)
parser.add_argument("--mlpipeline-ui-metadata-path", type=str)

args = parser.parse_args()
preprocess_data(args.load_data_path, args.preprocess_data_path, args.test_size, args.mlpipeline_ui_metadata_path)