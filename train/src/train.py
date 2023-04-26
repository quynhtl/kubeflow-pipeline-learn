import json
import os, pickle;
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras import layers
#import tensorboard
from tensorflow.keras.callbacks import TensorBoard


def modeling(preprocess_data_path: str,
            model_path: str,
            hidden_dim1: int,
            hidden_dim2:int,
            dropout: float,
            learning_rate: float,
            epochs: int,
            batch_size: int):

    # import Library

    #loading the train data
    with open(f'{preprocess_data_path}/train', 'rb') as f:
        train_data = pickle.load(f)

    # Separate the X_train from y_train.
    X_train, y_train = train_data

    #initializing the classifier model with its input, hidden and output layers
    if hidden_dim1 is None:
        hidden_dim1=56
    if hidden_dim2 is None:
        hidden_dim2=100
    DROPOUT=dropout
    model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters = hidden_dim1, kernel_size = (5,5),padding = 'Same',
                         activation ='relu'),
            tf.keras.layers.Dropout(DROPOUT),
            tf.keras.layers.Conv2D(filters = hidden_dim2, kernel_size = (3,3),padding = 'Same',
                         activation ='relu'),
            # tf.keras.layers.Dropout(DROPOUT),
            # tf.keras.layers.Conv2D(filters = hidden_dim2, kernel_size = (3,3),padding = 'Same',
            #              activation ='relu'),
            tf.keras.layers.Dropout(DROPOUT),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10, activation = "softmax")
            ])


    model.build(input_shape=(None,28,28,1))

    #Compiling the classifier model with Adam optimizer
    model.compile(optimizers.Adam(learning_rate=learning_rate),
              loss=SparseCategoricalCrossentropy(),
              metrics=[SparseCategoricalAccuracy(name='accuracy')])

    # model fitting

    tensorboard = TensorBoard(log_dir=f'{model_path}/logs')
    history = model.fit(np.array(X_train), np.array(y_train),
              validation_split=.1, epochs=epochs, batch_size=batch_size,
              callbacks=[tensorboard])

    # #loading the X_test and y_test
    # with open(f'{preprocess_data_path}/test', 'rb') as f:
    #     test_data = pickle.load(f)
    # # Separate the X_test from y_test.
    # X_test, y_test = test_data

    # # Evaluate the model and print the results
    # test_loss, test_acc = model.evaluate(np.array(X_test),  np.array(y_test), verbose=0)
    # print("Test_loss: {}, Test_accuracy: {} ".format(test_loss,test_acc))

    #creating the preprocess directory
    os.makedirs(model_path, exist_ok = True)

    #saving the model
    model.save(f'{model_path}/model.h5')

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--preprocess-data-path", type=str, default="preprocess_data_path")
parser.add_argument("--model-path", type=str)
parser.add_argument("--hidden-dim1", type=int, default=56)
parser.add_argument("--hidden-dim2", type=int, default=100)
parser.add_argument("--dropout", type=float, default=0.5)
parser.add_argument("--learning-rate", type=float, default=0.001)
parser.add_argument("--epochs", type=int, default=2)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--output-metadata-path", type=str)

args = parser.parse_args()


metadata = {
    'outputs' : [{
      'type': 'tensorboard',
      'source': f"{args.model_path}/logs",
    }]
  }

os.makedirs(os.path.dirname(args.output_metadata_path), exist_ok=True)
with open(args.output_metadata_path, 'w') as metadata_file:
    json.dump(metadata, metadata_file)


modeling(args.preprocess_data_path, args.model_path, args.hidden_dim1, args.hidden_dim2, args.dropout, args.learning_rate, args.epochs, args.batch_size)