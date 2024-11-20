import numpy as np
from typing import Tuple

def load_sequences(self, sequences_file: str, targets_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load the saved sequences and targets"""
    sequences = np.load(sequences_file, allow_pickle=True)
    targets = np.load(targets_file)
    return sequences, targets