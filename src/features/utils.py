import numpy as np
from typing import Tuple

def load_sequences(sequences_file: str, targets_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load the saved sequences and targets"""
    sequences = np.load(sequences_file, allow_pickle=True)
    targets = np.load(targets_file)
    return sequences, targets


def save_filtered_data(sequences: np.ndarray, 
                      targets: np.ndarray, 
                      sequences_file: str, 
                      targets_file: str):
    """
    Save filtered sequences and their corresponding targets to files
    """
    # Convert sequences to numpy array with object dtype to handle variable-length lists
    sequences_array = np.array(sequences, dtype=object)
    targets_array = np.array(targets)

    np.save(sequences_file, sequences_array)
    np.save(targets_file, targets_array)
    
    
