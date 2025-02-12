import numpy as np
from typing import Tuple
# meant to be imported by notebooks
# contains utility functions for data processing and visualization
# extended as experiments advance, meant to be backwards compatible
import os
import json
import pickle
import re
from functools import lru_cache
from uu import encode

import tensorflow as tf
import numpy as np
import sklearn.model_selection as sk_model_selection
import sklearn.utils as sk_utils
from keras.models import load_model

def load_sequences(self, sequences_file: str, targets_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load the saved sequences and targets"""
    sequences = np.load(sequences_file, allow_pickle=True)
    targets = np.load(targets_file)
    return sequences, targets


# define sklearn.utils.shuffle with a fixed random_state
def shuffle(*args, **kwargs):
    # remove random_state from kwargs
    kwargs.pop('random_state', None)
    return sk_utils.shuffle(*args, random_state=42, **kwargs)

# define sklearn.sklearn.model_selection.train_test_split with a fixed random_state
def train_test_split(*args, **kwargs):
    # remove random_state from kwargs
    kwargs.pop('random_state', None)
    return sk_model_selection.train_test_split(*args, random_state=42, **kwargs)


"""
general utilities
"""

def class_from_filename(filename):
    """
    Files have format: {label}_{subject_uuid}_{occurrence_id}.txt
    E.g. "adjkerntz_77C420A3-3B91-11E8-B8CE-15D78AC88FB6_2.txt".
    Note that the label may contain underscores (pwd_mkdb).
    """
    if filename.count('_') == 2:
        label, _, _ = filename.split('_')
    elif filename.count('_') == 3:
        label, tmp, _, _ = filename.split('_')
        label += '_' + tmp
    else:
        print(f'Unknown file name format: {filename}')
        raise ValueError()
    return label


"""
preprocessing utilities
notes:
    - the string 'None' is used to represent the absence of a value
    - the string 'Other' is used to represent a catch-all category
"""

def preprocess_path_to_tld(path):
    if path != 'None' and not path.startswith('/'):
        path = 'Other'
    if path != 'None' and path != 'Other':
        path = path.split('/')[1]
    return path

def preprocess_path_to_hash(path) -> int:
    if path == 'None':
        return 0
    return hash(path)

private_ip_ranges = [
    [ '10.0.0.0', '10.255.255.255' ],
    [ '100.64.0.0', '100.127.255.255' ],
    [ '172.16.0.0', '172.31.255.255' ],
    [ '192.0.0.0', '192.0.0.255' ],
    [ '192.168.0.0', '192.168.255.255' ],
    [ '198.18.0.0', '198.19.255.255' ],
]
loopback_ip_range = [ '127.0.0.0', '127.255.255.255' ]
local_network_ip_range = [ '0.0.0.0', '0.255.255.255' ]
internet_ip_ranges = [
    [ '224.0.0.0', '239.255.255.255' ],
    [ '240.0.0.0', '255.255.255.254' ],
]

def ip_to_int(ip):
    return sum([int(octet) * 256 ** i for i, octet in enumerate(reversed(ip.split('.')))])

def ip_in_range(ip, ip_range):
    """
    Checks if an IP is in a given range.
    A range is a list of two strings, each representing lower and upper bound of the range.
    """
    ip = ip_to_int(ip)
    lower_bound = ip_to_int(ip_range[0])
    upper_bound = ip_to_int(ip_range[1])
    return lower_bound <= ip <= upper_bound

def preprocess_ip(ip):
    if ip == 'None':
        return ip
    if ip == 'localhost':
        return 'Loopback'
    try:
        ip_to_int(ip)
    except ValueError:
        return 'Other'
    for range in private_ip_ranges:
        if ip_in_range(ip, range):
            return 'Private'
    if ip_in_range(ip, loopback_ip_range):
        return 'Loopback'
    if ip_in_range(ip, local_network_ip_range):
        return 'Local'
    for range in internet_ip_ranges:
        if ip_in_range(ip, range):
            return 'Internet'
    return 'Other'

ports_to_categories = {
    1: 'TCPMUX',
    22: 'SSH',
    25: 'SMTP',
    53: 'DNS',
    80: 'HTTP',
    143: 'IMAP',
    512: 'EXEC',
}

def preprocess_port(port):
    if port == 'None':
        return port
    port = int(port)
    return ports_to_categories.get(port, 'Other')


"""
data loading utilities
"""

class Preprocessor:
    """
    Class used to preprocess lines into vectors of preprocessed and vectorized values.
    """
    def __init__(self, enabled_features: list, **kwargs) -> None:
        """
        Initializes the preprocessor with the enabled features.
        List of valid features:
            - 'TYPE': event type
            - 'USERNAME': username
            - 'PRED_OBJ_TYPES': predicate object types (file, netobject, pipe, etc.)
            - 'PRED_OBJ_PATHS': predicate object paths
            - 'PRED_OBJ_NETINFO': predicate object network information
            - 'PRED_OBJ_PATHHASH': predicate object path hash
            - 'DELTA_TIME': time difference between events
            - 'PRED_OBJ_PATH_TOP_DIRS': most used files and directories. takes optional kwarg 'top_dirs' (default 20) which controls dimensionality
            - 'PRED_OBJ_PATH_AUTOENC': autoencoder for paths,
                - requires kwarg 'autoencoder_path' or 'encode_map'
                - 'autoencoder_path': path containing 'encoder.keras' and 'decoder.keras'
                - 'encode_map': dictionary mapping paths to pre-computed vector representations
            - 'SIZE': corresponds to size_long in event table
        """
        self.enabled_features = enabled_features

        self.event_types_map = {}
        self.users_map = {}
        self.filetypes_map = {'None': 0}
        self.path_map = {'None': 0}
        self.addr_map = {'None': 0}
        self.port_map = {'None': 0}

        self.encoder_mode_inference = False

        if 'PRED_OBJ_PATH_TOP_DIRS' in enabled_features:
            self.top_dir_map = {'None': 0, 'Other': 1}
            # get or set argument
            if 'top_dirs' in kwargs:
                self.top_dirs = kwargs['top_dirs']
            else:
                self.top_dirs = 20

            top_dirs_list = []
            # load list of top directories
            with open('top_dirs.txt', 'r') as f:
                for line in f:
                    top_dirs_list.append(line.strip('\n '))
            # add top directories to the map
            for i, dir in enumerate(top_dirs_list[:self.top_dirs]):
                self.top_dir_map[dir] = i + 2

        if 'PRED_OBJ_PATH_AUTOENC' in enabled_features:

            if 'autoencoder_path' in kwargs:
                self.encoder_mode_inference = True
                self.autoencoder_path = kwargs['autoencoder_path']
                self.encoder = load_model(os.path.join(self.autoencoder_path, 'encoder.keras'))
                self.decoder = load_model(os.path.join(self.autoencoder_path, 'decoder.keras'))
                with open(os.path.join(self.autoencoder_path, 'char_to_idx.json'), 'r') as f:
                    self.autoencoder_char_to_idx = json.load(f)
                with open(os.path.join(self.autoencoder_path, 'idx_to_char.json'), 'r') as f:
                    self.autoencoder_idx_to_char = json.load(f)

                self.autoencoder_none_path = self.encoder.predict(np.array([vectorize_datapoint('None', self.autoencoder_char_to_idx, fixed_length)]), verbose=0)[0]
            elif 'encode_map' in kwargs:
                self.encoder_mode_inference = False
                self.encode_map = pickle.load(open(kwargs['encode_map'], 'rb'))
                self.autoencoder_none_path = self.encode_map['None']


    def process(self, line: list) -> dict:
        """
        Processes a line from the dataset.
        Returns a dictionary with the processed values.
        Values are vectorized and mapped to integers.
        """
        line = line.strip('\n ').split(',')

        v = {}

        if 'TYPE' in self.enabled_features:
            assert len(line) >= 1
            v['TYPE'] = self.event_types_map.setdefault(line[0], len(self.event_types_map))

        if 'USERNAME' in self.enabled_features:
            assert len(line) >= 2
            v['USERNAME'] = self.users_map.setdefault(line[1], len(self.users_map))

        if 'PRED_OBJ_TYPES' in self.enabled_features:
            assert len(line) >= 4
            v['PRED_OBJ1_TYPE'] = self.filetypes_map.setdefault(line[2], len(self.filetypes_map))
            v['PRED_OBJ2_TYPE'] = self.filetypes_map.setdefault(line[3], len(self.filetypes_map))

        if 'PRED_OBJ_PATHS' in self.enabled_features:
            assert len(line) >= 6
            v['PRED_OBJ1_PATH'] = self.path_map.setdefault(preprocess_path_to_tld(line[4]), len(self.path_map))
            v['PRED_OBJ2_PATH'] = self.path_map.setdefault(preprocess_path_to_tld(line[5]), len(self.path_map))

        if 'PRED_OBJ_PATHHASH' in self.enabled_features:
            assert len(line) >= 6
            v['PRED_OBJ1_PATHHASH'] = preprocess_path_to_hash(line[4])
            v['PRED_OBJ2_PATHHASH'] = preprocess_path_to_hash(line[5])

        if 'PRED_OBJ_NETINFO' in self.enabled_features:
            assert len(line) >= 14
            v['PRED_OBJ1_LOCALIP'] = self.addr_map.setdefault(preprocess_ip(line[6]), len(self.addr_map))
            v['PRED_OBJ1_LOCALPORT'] = self.port_map.setdefault(preprocess_port(line[7]), len(self.port_map))
            v['PRED_OBJ1_REMOTEIP'] = self.addr_map.setdefault(preprocess_ip(line[8]), len(self.addr_map))
            v['PRED_OBJ1_REMOTEPORT'] = self.port_map.setdefault(preprocess_port(line[9]), len(self.port_map))
            v['PRED_OBJ2_LOCALIP'] = self.addr_map.setdefault(preprocess_ip(line[10]), len(self.addr_map))
            v['PRED_OBJ2_LOCALPORT'] = self.port_map.setdefault(preprocess_port(line[11]), len(self.port_map))
            v['PRED_OBJ2_REMOTEIP'] = self.addr_map.setdefault(preprocess_ip(line[12]), len(self.addr_map))
            v['PRED_OBJ2_REMOTEPORT'] = self.port_map.setdefault(preprocess_port(line[13]), len(self.port_map))

        if 'DELTA_TIME' in self.enabled_features:
            assert len(line) >= 15
            v['DELTA_TIME'] = int(line[14])

        if 'PRED_OBJ_PATH_TOP_DIRS' in self.enabled_features:
            assert len(line) >= 6
            # check if paths are None
            for path, key in zip(line[4:6], ['PRED_OBJ_PATH1_TOP_DIR', 'PRED_OBJ_PATH2_TOP_DIR']):
                # if path is None, set feature to 0
                if path == 'None':
                    v[key] = 0
                # if path is in the map in its exact form
                elif path in self.top_dir_map:
                    v[key] = self.top_dir_map[path]
                # path is not in the map, try to find the closest match.
                # meaning if we have something like /home/anon/file.txt, we want to find /home or better yet /home/anon
                else:
                    # find the longest match
                    longest_match = 0
                    matching_dir = None
                    for dir in self.top_dir_map.keys():
                        if path.startswith(dir) and dir.count('/') > longest_match:
                            longest_match = dir.count('/')
                            matching_dir = dir
                    # if no match was found, set feature to 'Other'
                    if longest_match == 0:
                        v[key] = 1
                    # if a match was found, set feature to the match
                    else:
                        v[key] = self.top_dir_map[matching_dir]

        if 'PRED_OBJ_PATH_AUTOENC' in self.enabled_features:
            assert len(line) >= 6
            # encode the paths
            for path, key in zip(line[4:6], ['PRED_OBJ1_PATH_AUTOENC', 'PRED_OBJ2_PATH_AUTOENC']):
                if path == 'None':
                    v[key] = self.autoencoder_none_path
                    continue

                if self.encoder_mode_inference:
                    path_vectorized = vectorize_datapoint(path, self.autoencoder_char_to_idx, fixed_length)
                    path_vectorized_tuple = array_to_tuple(path_vectorized)  # Convert to tuple
                    path_encoded = encode_path_cached(path_vectorized_tuple, self.encoder)  # Use the cached function
                    v[key] = path_encoded
                else:
                    v[key] = self.encode_map.get(path, self.autoencoder_none_path)

            assert 'PRED_OBJ1_PATH_AUTOENC' in v
            assert 'PRED_OBJ2_PATH_AUTOENC' in v

        if 'SIZE' in self.enabled_features:
            assert len(line) >= 16
            v['SIZE'] = float(line[15]) if line[15] != 'None' else 0

        return v





"""
autoencoder utilities
"""
def preprocess_path(path: str) -> str:
    path = path.lower()
    path = re.sub(r'[^a-z0-9/._]', '', path)
    return path

fixed_length = 50

def vectorize_data(X, char_to_idx, fixed_length):
    X_vec = np.zeros((len(X), fixed_length), dtype=np.int32)
    for i, path in enumerate(X):
        for j, char in enumerate(path):
            if j >= fixed_length:
                break
            X_vec[i, j] = char_to_idx[char]
    return X_vec

def vectorize_datapoint(path: str, char_to_idx: dict, fixed_length: int) -> np.ndarray:
    path = preprocess_path(path)
    X_vec = np.zeros((fixed_length), dtype=np.int32)
    for j, char in enumerate(path):
        if j >= fixed_length:
            break
        X_vec[j] = char_to_idx[char]
    return X_vec

def vectorized_to_string(X, idx_to_char):
    return [''.join([idx_to_char[idx] for idx in path]) for path in X]

def output_to_string(output: np.ndarray, idx_to_char: dict) -> str:
    output_argmax = np.argmax(output, axis=-1)
    return vectorized_to_string(output_argmax, idx_to_char)


def array_to_tuple(arr):
    return tuple(arr)

@lru_cache(maxsize=500000)
def encode_path_cached(path_vectorized_tuple, encoder):
    path_vectorized = np.array(path_vectorized_tuple)
    path_encoded = encoder.predict(np.array([path_vectorized]), verbose=0)[0]
    return path_encoded



import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def analyze_size_distribution(X_train, X_val, X_test):
    """
    Analysiert die Verteilung der SIZE-Werte in den Datensätzen
    """
    # Sammle alle SIZE-Werte (Index 10)
    sizes_train = []
    sizes_val = []
    sizes_test = []
    
    for sequence in X_train:
        sizes_train.extend(sequence[:, 10])
    for sequence in X_val:
        sizes_val.extend(sequence[:, 10])
    for sequence in X_test:
        sizes_test.extend(sequence[:, 10])
    
    # Konvertiere zu numpy arrays
    sizes_train = np.array(sizes_train)
    sizes_val = np.array(sizes_val)
    sizes_test = np.array(sizes_test)
    
    # Berechne Statistiken
    print("\nTraining Set Statistiken:")
    print(f"Anzahl der Werte: {len(sizes_train)}")
    print(f"Mittelwert: {np.mean(sizes_train):.2f}")
    print(f"Standardabweichung: {np.std(sizes_train):.2f}")
    print(f"Median: {np.median(sizes_train):.2f}")
    print(f"Min: {np.min(sizes_train):.2f}")
    print(f"Max: {np.max(sizes_train):.2f}")
    
    # Erstelle Plots
    plt.figure(figsize=(15, 10))
    
    
    # Log-Scale Histogram
    plt.subplot(2, 2, 2)
    plt.hist(sizes_train, bins=50, alpha=0.7, log=True)
    plt.title('Log-Scale Histogram der SIZE-Werte')
    plt.xlabel('SIZE')
    plt.ylabel('Log Häufigkeit')
    
    
    plt.tight_layout()
    plt.show()
    
    # Zusätzliche Analysen
    print("\nWertebereiche:")
    ranges = [(float('-inf')  , 0), (0, 1000), (1000, 10000), (10000, 100000), (100000, float('inf'))]
    for start, end in ranges:
        count = np.sum((sizes_train >= start) & (sizes_train < end))
        percentage = (count / len(sizes_train)) * 100
        print(f"SIZE zwischen {start} und {end}: {count} ({percentage:.2f}%)")
    
    # Test auf Normalverteilung
    _, p_value = stats.normaltest(sizes_train)
    print(f"\nTest auf Normalverteilung p-value: {p_value}")
    if p_value < 0.05:
        print("Die Verteilung ist wahrscheinlich nicht normal (p < 0.05)")
    else:
        print("Die Verteilung könnte normal sein (p >= 0.05)")
        
        
from sklearn.preprocessing import StandardScaler


def normalize_size_feature(X_train, X_val, X_test):
    """
    Normalisiert nur das SIZE Feature
    SIZE befindet sich an Index 10 in den Sequenzen
    """
    # Initialisiere Scaler
    size_scaler = StandardScaler()
    
    # Sammle alle SIZE Werte aus dem Training Set
    sizes_train = []
    for sequence in X_train:
        sizes_train.extend(sequence[:, 10].reshape(-1, 1))
    
    # Fit Scaler nur auf Trainingsdaten
    size_scaler.fit(sizes_train)
    
    # Transformationsfunktion für eine einzelne Sequenz
    def transform_sequence(sequence):
        seq_transformed = sequence.copy()
        seq_transformed[:, 10] = size_scaler.transform(sequence[:, 10].reshape(-1, 1)).flatten()
        return seq_transformed
    
    # Transformiere alle Datensätze
    X_train_norm = [transform_sequence(seq) for seq in X_train]
    X_val_norm = [transform_sequence(seq) for seq in X_val]
    X_test_norm = [transform_sequence(seq) for seq in X_test]
    
    return X_train_norm, X_val_norm, X_test_norm