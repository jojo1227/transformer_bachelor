import numpy as np
from typing import List, Dict, Tuple
from collections import Counter

class Encoder:
    def __init__(self, max_sequence_length: int = 512):
        self.max_sequence_length = max_sequence_length
        self.event_to_idx: Dict[str, int] = {}
        self.idx_to_event: Dict[int, str] = {}
        self.executable_to_idx: Dict[str, int] = {}
        self.idx_to_executable: Dict[int, str] = {}
        self.pad_token_id = 0
        self.vocab_size = 0
        
    def build_vocabulary(self, sequences: List[List[str]], executables: List[str]):
        """Erstellt das Vokabular aus den Event-Sequenzen"""
        # Zähle alle einzigartigen Events
        event_counter = Counter([event for seq in sequences for event in seq])
        unique_events = sorted(event_counter.keys())
        
        # Erstelle Event-Mappings (0 ist für Padding reserviert)
        self.event_to_idx = {event: idx + 1 for idx, event in enumerate(unique_events)}
        self.idx_to_event = {idx: event for event, idx in self.event_to_idx.items()}
        self.event_to_idx['<PAD>'] = 0
        self.idx_to_event[0] = '<PAD>'
        
        
        # Erstelle Executable-Mappings
        unique_executables = sorted(set(executables))
        self.executable_to_idx = {exe: idx for idx, exe in enumerate(unique_executables)}
        self.idx_to_executable = {idx: exe for exe, idx in self.executable_to_idx.items()}
        
        
        self.vocab_size = len(self.event_to_idx)
        
        
        return self
    

    
    def encode_sequences(self, sequences: List[List[str]], targets: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Konvertiert String-Sequenzen in numerische Arrays"""
        
        # Konvertiere Events zu Indices
        encoded_sequences = [[self.event_to_idx[event] for event in seq] for seq in sequences]
        
        # Konvertiere Targets zu Indices
        encoded_targets = np.array([self.executable_to_idx[exe] for exe in targets])
        
        return encoded_sequences, encoded_targets
    
        # TODO Datensatz verfälscht? Random Teil der Seuqeunz nehmen, da sonst eine Sequenz oversampled wird. 
        # TODO ist die Sequenzlänge zu kurz? Zufällig

    
    def split_long_sequences(
        self, 
        sequences: List[List[int]], 
        targets: List[int], 
        overlap_ratio: float = 0.25
    ) -> Tuple[List[List[int]], List[int]]:
        """
        Teilt Sequenzen auf, die länger als max_sequence_length sind.
        Verwendet einen konfigurierbaren überlappenden Ansatz zur Kontextbewahrung.
        
        :param sequences: Liste der ursprünglichen Sequenzen
        :param targets: Zugehörige Ziellabels
        :param overlap_ratio: Anteil der Überlappung (0 bis 1)
        :return: Tuple aus aufgeteilten Sequenzen und Targets
        """
        new_sequences = []
        new_targets = []
        
        # Berechne Überlappung in absoluten Werten
        overlap = int(self.max_sequence_length * max(0, min(overlap_ratio, 1)))
        
        for seq, target in zip(sequences, targets):
            if len(seq) <= self.max_sequence_length:
                # Sequenz ist kurz genug, füge sie direkt hinzu
                new_sequences.append(seq)
                new_targets.append(target)
            else:
                # Teile lange Sequenz in überlappende Teilsequenzen
                start = 0
                while start < len(seq):
                    # Extrahiere Teilsequenz
                    end = start + self.max_sequence_length
                    sub_sequence = seq[start:end]
                    
                    # Füge nur Teilsequenzen hinzu, die lang genug sind
                    if len(sub_sequence) >= self.max_sequence_length // 2:
                        new_sequences.append(sub_sequence)
                        new_targets.append(target)
                    
                    # Verschiebe start-Position unter Berücksichtigung der Überlappung
                    start = end - overlap
        
        return new_sequences, new_targets
    
    def pad_sequences(self, sequences: List[List[int]]) -> np.ndarray:
        """
        Bringt alle Sequenzen auf die gleiche Länge durch Padding.
        
        Args:
            sequences: Liste von numerischen Sequenzen
            padding: 'post' für Padding am Ende, 'pre' für Padding am Anfang
            
        Returns:
            np.ndarray mit gepadten Sequenzen der Form (n_sequences, max_sequence_length)
        """
        n_sequences = len(sequences)
        padded_sequences = np.zeros((n_sequences, self.max_sequence_length), dtype=np.int32)
        
        for idx, seq in enumerate(sequences):
            if len(seq) > self.max_sequence_length:
                # darf nicht passieren
                print("Sequenz ist länger als sie sein darf")
            else:
                padded_sequences[idx, :len(seq)] = seq
                padded_sequences[idx, len(seq):] = self.pad_token_id
              
                    
        return padded_sequences