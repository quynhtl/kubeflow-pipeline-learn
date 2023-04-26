import wget
import zipfile
import os

import argparse



def download_data(download_link:str, data_path: str):

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    wget.download(download_link.format(file='train'), f'{data_path}/train_csv.zip')
    wget.download(download_link.format(file='test'), f'{data_path}/test_csv.zip')
    with zipfile.ZipFile(f"{data_path}/train_csv.zip","r") as zip_ref:
        zip_ref.extractall(data_path)

    with zipfile.ZipFile(f"{data_path}/test_csv.zip","r") as zip_ref:
        zip_ref.extractall(data_path)

parser = argparse.ArgumentParser(description='Download data')
parser.add_argument('--data-path', type=str, default='./data')
parser.add_argument('--download-link', type=str, default='https://github.com/kubeflow/examples/blob/master/digit-recognition-kaggle-competition/data/{file}.csv.zip?raw=true')

args = parser.parse_args()

download_data(args.download_link, args.data_path)

