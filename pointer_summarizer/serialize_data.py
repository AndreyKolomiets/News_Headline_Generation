import sys
import os
import hashlib
import struct
import subprocess
import collections
from tensorflow.core.example import example_pb2
from tqdm import tqdm
from multiprocessing import Pool
import argparse
from utils import load_data, parse_source


parser = argparse.ArgumentParser()
parser.add_argument('--path_to_data', type=str, default='/home/jovyan/work/_Headline_generation/ria.json')
parser.add_argument('--n_cores', type=int, default=10)
args = parser.parse_args()

path_to_data = args.path_to_data
path_to_tokenized = '/'.join(path_to_data.split('/')[:-1]) + '/tokenized'

texts, titles = load_data(path_to_data)
pool = Pool(10)
parsed = pool.map(parse_source, texts)


if __name__ == '__main__':
    if not os.path.exists(path_to_tokenized):
        os.makedirs(path_to_tokenized)
