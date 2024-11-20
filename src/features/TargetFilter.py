from collections import Counter
import numpy as np

class TargetFilter:
    def __init__(self, unusable_threshold: int = 50, rare_threshold: int = 100):
        self.unusable_threshold = unusable_threshold
        self.rare_threshold = rare_threshold
        self.total_classes = 0
        self.unusable_classes = []
        self.rare_classes = []
        self.usable_classes = []
        
    def filter_data(self, targets: np.ndarray, sequences: np.ndarray):
        """
        Filter both targets and sequences based on target frequency
        
        Args:
            targets (np.ndarray): Array of target labels
            sequences (np.ndarray): Array of corresponding sequences
            
        Returns:
            tuple: (filtered_sequences, filtered_targets, filtering_info)
        """
        # Count occurrences of each target
        target_counts = Counter(targets)
        
        self.total_classes = len(target_counts)
        
        # Reset lists
        self.unusable_classes = []
        self.rare_classes = []
        self.usable_classes = []
        
        # Categorize targets
        for label, count in target_counts.items():
            if count < self.unusable_threshold:
                self.unusable_classes.append(label)
            elif count < self.rare_threshold:
                self.rare_classes.append(label)
                self.usable_classes.append(label)
            else:
                self.usable_classes.append(label)
        
        # Create a mask for usable classes
        mask = np.isin(targets, self.usable_classes)
        
        # Apply mask to both sequences and targets
        filtered_sequences = sequences[mask]
        filtered_targets = targets[mask]
        
        # Prepare filtering information
        filtering_info = {
            'total_classes': self.total_classes,
            'unusable_classes': self.unusable_classes,
            'rare_classes': self.rare_classes,
            'usable_classes': self.usable_classes,
            'original_samples': len(targets),
            'filtered_samples': len(filtered_targets)
        }
        
        return filtered_sequences, filtered_targets, filtering_info
    
    def oversample_rare_classes(self, 
                              targets: np.ndarray, 
                              sequences: np.ndarray, 
                              target_sample_count: int = 1000):
        """
        Oversample rare classes to balance the dataset
        
        Args:
            targets (np.ndarray): Original target labels
            sequences (np.ndarray): Corresponding sequences
            target_sample_count (int): Desired sample count for rare classes
        
        Returns:
            tuple: (oversampled_sequences, oversampled_targets)
        """
        # Create lists to store oversampled data
        oversampled_sequences = list(sequences)
        oversampled_targets = list(targets)
        
        # Oversample rare classes
        for rare_class in self.rare_classes:
            # Find indices of rare class
            rare_indices = np.where(targets == rare_class)[0]
            
            # If rare class has fewer samples than target, oversample
            if len(rare_indices) < target_sample_count:
                # Randomly sample with replacement to reach target count
                additional_indices = np.random.choice(
                    rare_indices, 
                    size=target_sample_count - len(rare_indices), 
                    replace=True
                )
                
                # Add oversampled sequences and targets
                for idx in additional_indices:
                    oversampled_sequences.append(sequences[idx])
                    oversampled_targets.append(rare_class)
        
        return np.array(oversampled_sequences, dtype=object), np.array(oversampled_targets)