import json
import torch
from torch.utils.data import Dataset
import numpy as np

# TODO Attention mask and padding here
class CustomDataset(Dataset):
    def __init__(self, sequences_path: str, targets_path: str):
        """
        Lädt die Sequenzen und die Targets aus den angegebenen Dateien.
        """
       
        self.sequences = np.load(sequences_path, allow_pickle=True)
        self.targets = np.load(targets_path)
        
        self.len = len(self.sequences)  # Länge des Datasets

    def __len__(self):
        """Gibt die Anzahl der Beispiele im Dataset zurück."""
        return self.len

    def __getitem__(self, idx):
        """Gibt ein Beispiel aus dem Dataset zurück."""
        sequence = np.array(self.sequences[idx], dtype=np.int64)

        sequence = torch.tensor(sequence, dtype=torch.long)  # Sequenz als LongTensor
        target = torch.tensor(self.targets[idx], dtype=torch.long)  # Label als LongTensor
        return sequence, target
