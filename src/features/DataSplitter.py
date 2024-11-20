import numpy as np
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt

class DataSplitter:
    def __init__(self, sequences_file: str, targets_file: str):
        """
        Initialisiert den Splitter mit Sequenzen und Targets

        Args:
            sequences_file (str): Pfad zur Numpy-Datei mit Sequenzen
            targets_file (str): Pfad zur Numpy-Datei mit Targets
        """
        self.sequences = np.load(sequences_file, allow_pickle=True)
        self.targets = np.load(targets_file)

    def analyze_target_distribution(self, targets):
        """
        Analysiert die Verteilung der Targets

        Args:
            targets (np.ndarray): Array mit Targets

        Returns:
            dict: Verteilung der Targets
        """
        unique, counts = np.unique(targets, return_counts=True)
        distribution = dict(zip(unique, counts / len(targets) * 100))
        return distribution

    def split_data(
        self, 
        test_size: float = 0.15, 
        val_size: float = 0.15, 
        random_state: int = 42
    ):
        """
        Teilt Daten mit Stratifizierung

        Args:
            test_size (float): Anteil der Testdaten
            val_size (float): Anteil der Validierungsdaten
            random_state (int): Seed für Reproduzierbarkeit

        Returns:
            Tupel mit (train_sequences, test_sequences, val_sequences,
                       train_targets, test_targets, val_targets)
        """
        # Ursprüngliche Verteilung ausgeben
        print("Ursprüngliche Targetverteilung:")
        original_dist = self.analyze_target_distribution(self.targets)
        for target, percentage in original_dist.items():
            print(f"Target {target}: {percentage:.2f}%")

        # Erster Split: Train+Validation und Test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            self.sequences, 
            self.targets, 
            test_size=test_size, 
            stratify=self.targets,  # Stratifizierung aktivieren
            random_state=random_state
        )

        # Zweiter Split: Train und Validation
        val_relative_size = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, 
            y_train_val, 
            test_size=val_relative_size, 
            stratify=y_train_val,  # Stratifizierung für zweiten Split
            random_state=random_state
        )

        # Verteilung der gesplitteten Datensätze ausgeben
        print("\nTargetverteilung nach Split:")
        print("Test-Set:")
        test_dist = self.analyze_target_distribution(y_test)
        for target, percentage in test_dist.items():
            print(f"Target {target}: {percentage:.2f}%")

        print("\nValidation-Set:")
        val_dist = self.analyze_target_distribution(y_val)
        for target, percentage in val_dist.items():
            print(f"Target {target}: {percentage:.2f}%")

        print("\nTraining-Set:")
        train_dist = self.analyze_target_distribution(y_train)
        for target, percentage in train_dist.items():
            print(f"Target {target}: {percentage:.2f}%")

        return X_train, X_test, X_val, y_train, y_test, y_val

    def save_splits(
        self, 
        output_dir: str, 
        test_size: float = 0.2, 
        val_size: float = 0.2, 
        random_state: int = 42
    ):
        """
        Speichert die aufgeteilten Daten in Dateien

        Args:
            output_dir (str): Verzeichnis zum Speichern der Splits
            test_size (float): Anteil der Testdaten
            val_size (float): Anteil der Validierungsdaten
            random_state (int): Seed für Reproduzierbarkeit
        """
        # Stelle sicher, dass Ausgabeverzeichnis existiert
        os.makedirs(output_dir, exist_ok=True)

        # Daten aufteilen
        X_train, X_test, X_val, y_train, y_test, y_val = self.split_data(
            test_size, val_size, random_state
        )
        
        #Plotten der Targetverteilung
        self.plot_target_distribution(train_targets=y_train,test_targets=y_test,val_targets=y_val)

        # Speichern der Splits
        np.save(os.path.join(output_dir, 'train_sequences.npy'), X_train)
        np.save(os.path.join(output_dir, 'test_sequences.npy'), X_test)
        np.save(os.path.join(output_dir, 'val_sequences.npy'), X_val)

        np.save(os.path.join(output_dir, 'train_targets.npy'), y_train)
        np.save(os.path.join(output_dir, 'test_targets.npy'), y_test)
        np.save(os.path.join(output_dir, 'val_targets.npy'), y_val)

        print(f"\nDaten gesplittet und gespeichert in {output_dir}")
        print(f"Train Samples: {len(X_train)}")
        print(f"Test Samples: {len(X_test)}")
        print(f"Validation Samples: {len(X_val)}")
        
    def plot_target_distribution(
        self,
        train_targets: np.ndarray, 
        test_targets: np.ndarray, 
        val_targets: np.ndarray, 
        output_dir: str = None,
        title: str = "Target Distribution Across Sets"
    ):
        """
        Visualisiert die Targetverteilung in verschiedenen Sets

        Args:
            train_targets (np.ndarray): Targets im Trainingset
            test_targets (np.ndarray): Targets im Testset
            val_targets (np.ndarray): Targets im Validationset
            output_dir (str, optional): Verzeichnis zum Speichern des Plots
            title (str, optional): Titel des Plots
        """
        plt.style.use('ggplot')
        
        # Definiere die Sets
        sets = [
            ('Training Set', train_targets),
            ('Test Set', test_targets),
            ('Validation Set', val_targets)
        ]

        # Erstelle separate Plots für jedes Set
        for set_name, targets in sets:
            # Neue Figure für jedes Set
            plt.figure(figsize=(20, 6))
            
            # Unique Werte und ihre Häufigkeiten
            unique, counts = np.unique(targets, return_counts=True)
            percentages = counts / len(targets) * 100
            
            # Bar Plot mit erhöhtem Abstand zwischen Bars
            bars = plt.bar(unique.astype(str), percentages, width=0.8)
            plt.title(f"{set_name} Distribution", fontsize=14, pad=20)
            plt.xlabel('Target', fontsize=12)
            plt.ylabel('Percentage (%)', fontsize=12)
                
            # Prozentwerte auf die Bars schreiben
            for bar, percentage in zip(bars, percentages):
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width()/2., 
                    height,
                    f'{percentage:.1f}%',
                    ha='center',
                    va='bottom',
                    fontsize=10
                )

            # Layout anpassen
            plt.tight_layout()

            # Speichern oder Anzeigen
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                plt.savefig(os.path.join(output_dir, f'{set_name.lower().replace(" ", "_")}_distribution.png'), 
                        dpi=300, 
                        bbox_inches='tight')
                plt.close()
            else:
                plt.show()
