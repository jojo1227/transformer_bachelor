import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from typing import Dict, List, Optional
import matplotlib.pyplot as plt

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: torch.optim.Optimizer = torch.optim.Adam,
        criterion: torch.nn.Module = nn.CrossEntropyLoss(),
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Args:
            model: Das zu trainierende Modell
            train_loader: DataLoader für Trainingsdaten
            val_loader: DataLoader für Validierungsdaten (optional)
            learning_rate: Lernrate für den Optimizer
            device: Device für das Training ("cuda" oder "cpu")
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Optimizer und Loss
        self.optimizer = optimizer
        self.criterion = criterion
        
        # Für Tracking der Metriken
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.train_accuracies: List[float] = []
        self.val_accuracies: List[float] = []
        
    def train_epoch(self) -> Dict[str, float]:
        """Trainiert das Modell für eine Epoche."""
        self.model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        # Progress bar
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for sequences, labels in progress_bar:
            # Daten auf device verschieben
            sequences = sequences.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad() #Gradienten werden zurückgesetzt
            outputs = self.model(sequences)  # Kein attention_mask mehr nötig
            
            # Loss berechnen
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Metriken berechnen
            total_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)
            
            # Progress bar updaten
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{correct_predictions/total_predictions:.4f}"
            })
            
        # Durchschnittliche Metriken berechnen
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct_predictions / total_predictions
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validiert das Modell auf dem Validierungsset."""
        if not self.val_loader:
            return {}
        
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        progress_bar = tqdm(self.val_loader, desc="Validating")

        
        for sequences, targets in progress_bar:
            # Daten auf device verschieben
            sequences = sequences.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.model(sequences)
            loss = self.criterion(outputs, labels)
            
            # Metriken berechnen
            total_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)
        
        # Durchschnittliche Metriken berechnen
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct_predictions / total_predictions
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def train(
        self,
        num_epochs: int,
        save_path: Optional[str] = None,
        early_stopping_patience: int = 5
    ):
        """
        Trainiert das Modell für mehrere Epochen.
        
        Args:
            num_epochs: Anzahl der Epochen
            save_path: Pfad zum Speichern des besten Modells
            early_stopping_patience: Anzahl der Epochen zu warten bevor Training gestoppt wird
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Training
            train_metrics = self.train_epoch()
            self.train_losses.append(train_metrics['loss'])
            self.train_accuracies.append(train_metrics['accuracy'])
            
            # Validation
            if self.val_loader:
                val_metrics = self.validate()
                self.val_losses.append(val_metrics['loss'])
                self.val_accuracies.append(val_metrics['accuracy'])
                
                print(f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}")
                print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
                
                # Early Stopping & Model Saving
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    patience_counter = 0
                    if save_path:
                        torch.save(self.model.state_dict(), save_path)
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        print(f"\nEarly stopping triggered after epoch {epoch+1}")
                        break
            else:
                print(f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}")
    
    def plot_metrics(self):
        """Plottet die Trainings- und Validierungsmetriken."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss Plot
        ax1.plot(self.train_losses, label='Training Loss')
        if self.val_loader:
            ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_title('Loss over epochs')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Accuracy Plot
        ax2.plot(self.train_accuracies, label='Training Accuracy')
        if self.val_loader:
            ax2.plot(self.val_accuracies, label='Validation Accuracy')
        ax2.set_title('Accuracy over epochs')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()