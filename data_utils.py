from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import scipy.sparse as sp
import random
import pdb

# For automatic dataset downloading
from urllib.request import urlopen
from zipfile import ZipFile
import shutil
import os.path
from tqdm import tqdm
try:
    from BytesIO import BytesIO
except ImportError:
    from io import BytesIO

# Map data to proper indices in case they are not in a continues [0, N) range
def map_data(data):
    uniq = list(set(data))

    id_dict = {old: new for new, old in enumerate(sorted(uniq))}
    data = np.array([id_dict[x] for x in data])
    n = len(uniq)

    return data, id_dict, n

# Downloads dataset if files are not present.
def download_dataset(dataset, files, data_dir):
    if not np.all([os.path.isfile(data_dir + f) for f in files]):
        url = "http://files.grouplens.org/datasets/movielens/" + dataset.replace('_', '-') + '.zip'
        request = urlopen(url)

        print('Downloading %s dataset' % dataset)

        if dataset in ['ml_100k', 'ml_1m']:
            target_dir = 'raw_data/' + dataset.replace('_', '-')
        elif dataset == 'ml_10m':
            target_dir = 'raw_data/' + 'ml-10M100K'
        else:
            raise ValueError('Invalid dataset option %s' % dataset)

        with ZipFile(BytesIO(request.read())) as zip_ref:
            zip_ref.extractall('raw_data/')

        os.rename(target_dir, data_dir)

# Loads dataset and creates adjacency matrix and feature matrix
def load_data(fname, seed=1234, verbose=True):
    u_features = None
    v_features = None

    print('Loading dataset', fname)

    data_dir = 'raw_data/' + fname

    if fname == 'ml_100k':

        # Check if files exist and download otherwise
        files = ['/u.data', '/u.item', '/u.user']

        download_dataset(fname, files, data_dir)

        sep = '\t'
        filename = data_dir + files[0]

        dtypes = {
            'u_nodes': np.int32, 'v_nodes': np.int32,
            'ratings': np.float32, 'timestamp': np.float64}

        data = pd.read_csv(
            filename, sep=sep, header=None,
            names=['u_nodes', 'v_nodes', 'ratings', 'timestamp'], dtype=dtypes)

        data_array = data.values.tolist()
        random.seed(seed)
        random.shuffle(data_array)
        data_array = np.array(data_array)

        u_nodes_ratings = data_array[:, 0].astype(dtypes['u_nodes'])
        v_nodes_ratings = data_array[:, 1].astype(dtypes['v_nodes'])
        ratings = data_array[:, 2].astype(dtypes['ratings'])

        u_nodes_ratings, u_dict, num_users = map_data(u_nodes_ratings)
        v_nodes_ratings, v_dict, num_items = map_data(v_nodes_ratings)

        u_nodes_ratings, v_nodes_ratings = u_nodes_ratings.astype(np.int64), v_nodes_ratings.astype(np.int32)
        ratings = ratings.astype(np.float64)

    return num_users, num_items, u_nodes_ratings, v_nodes_ratings, ratings
