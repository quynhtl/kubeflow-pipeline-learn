import tensorflow as tf
from tensorflow import keras
import pickle
import numpy as np
import json
import os

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--preprocess-data-path", type=str, default="preprocess_data_path")
parser.add_argument("--model-path", type=str)
parser.add_argument("--output-metadata-path", type=str)

args = parser.parse_args()

model = keras.models.load_model(f'{args.model_path}/model.h5')
with open(f'{args.preprocess_data_path}/test', 'rb') as f:
        test_data = pickle.load(f)

X_test, y_test = test_data




def eval(model, X_test, y_test):
    test_loss, test_acc = model.evaluate(np.array(X_test),  np.array(y_test), verbose=0)
    #convert from numpy to python float
    test_loss = float(test_loss)
    test_acc = float(test_acc)

    metrics = {
        'metrics': [{
            'name': 'test-loss',
            'numberValue': test_loss,
            'format': "RAW"
        }, {
            'name': 'test-accuracy',
            'numberValue': test_acc,
            'format': "PERCENTAGE"
        }]
    }
    return json.dumps(metrics)


outputs = eval(model, X_test, y_test)
os.makedirs(os.path.dirname(args.output_metadata_path), exist_ok=True)
with open(args.output_metadata_path, 'w') as f:
    f.write(outputs)

