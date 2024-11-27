import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import wandb
from datetime import datetime
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
        Initialize Trainer with model, data loaders, and training parameters.
        
        Args:
            model: The model to be trained
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation data
            optimizer: Optimizer for training
            criterion: Loss function
            device: Training device (cuda or cpu)
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Optimizer and Loss
        self.optimizer = optimizer
        self.criterion = criterion
        
        # Metrics tracking
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.train_accuracies: List[float] = []
        self.val_accuracies: List[float] = []
        
        # Additional metrics tracking
        self.train_precisions: List[float] = []
        self.train_recalls: List[float] = []
        self.train_f1_scores: List[float] = []
        self.val_precisions: List[float] = []
        self.val_recalls: List[float] = []
        self.val_f1_scores: List[float] = []
    
    def train_epoch(self) -> Dict[str, float]:
        
            
        """Train the model for one epoch and collect detailed metrics."""
        self.model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        # Lists to collect predictions and labels for detailed metrics
        all_predictions = []
        all_labels = []
        
        # Progress bar
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for sequences, labels, attention_masks in progress_bar:
            # Move data to device
            sequences = sequences.to(self.device)
            labels = labels.to(self.device)
            attention_masks = attention_masks.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(sequences, attention_masks)  
            
            # Compute loss
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Compute metrics
            total_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)
            
            # Collect predictions for detailed metrics
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{correct_predictions/total_predictions:.4f}"
            })
        
        # Compute average metrics
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct_predictions / total_predictions
        
        # Compute additional metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        confusion_mat = confusion_matrix(all_labels, all_predictions)
        
        return {
            'train/loss': avg_loss, 
            'train/accuracy': accuracy,
            'train/precision': precision,
            'train/recall': recall,
            'train/f1_score': f1,
            'train/confusion_matrix': confusion_mat
        }
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model on validation dataset."""
        if not self.val_loader:
            return {}
        
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        # Lists to collect predictions and labels for detailed metrics
        all_predictions = []
        all_labels = []
        
        progress_bar = tqdm(self.val_loader, desc="Validating")
        
        for sequences, targets, attention_masks in progress_bar:
            # Move data to device
            sequences = sequences.to(self.device)
            labels = targets.to(self.device)
            attention_masks = attention_masks.to(self.device)
            
            # Forward pass
            outputs = self.model(sequences, attention_masks)
            loss = self.criterion(outputs, labels)
            
            # Compute metrics
            total_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)
            
            # Collect predictions for detailed metrics
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        # Compute average metrics
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct_predictions / total_predictions
        
        # Compute additional metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        confusion_mat = confusion_matrix(all_labels, all_predictions)
        
        return {
            'val/loss': avg_loss, 
            'val/accuracy': accuracy,
            'val/precision': precision,
            'val/recall': recall,
            'val/f1_score': f1,
            'val/confusion_matrix': confusion_mat
        }
    
    def train(
        self,
        num_epochs: int,
        save_path: Optional[str] = None,
        early_stopping_patience: int = 5
    ):
        """
        Train the model for multiple epochs with early stopping.
        
        Args:
            num_epochs: Number of training epochs
            save_path: Path to save the best model
            early_stopping_patience: Epochs to wait before stopping training
        """
        
        wandb.init(
            # Set the project where this run will be logged
            project="basic-modell",
            # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
            name=f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            # Track hyperparameters and run metadata
            config={
            "learning_rate": 0.001,
            "architecture": "TRANSFORMER ENCODER",
            "dataset": "cadets-e3-v2",
            "epochs": 100,
            "dropout" : 0.1
        })
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Training
            train_metrics = self.train_epoch()
            self.train_losses.append(train_metrics['train/loss'])
            self.train_accuracies.append(train_metrics['train/accuracy'])
            self.train_precisions.append(train_metrics.get('train/precision', 0))
            self.train_recalls.append(train_metrics.get('train/recall', 0))
            self.train_f1_scores.append(train_metrics.get('train/f1_score', 0))
            
            # Detailed training metrics output
            print(f"Train Loss: {train_metrics['train/loss']:.4f}")
            print(f"Train Accuracy: {train_metrics['train/accuracy']:.4f}")
            print(f"Train Precision: {train_metrics.get('train/precision', 'N/A'):.4f}")
            print(f"Train Recall: {train_metrics.get('train/recall', 'N/A'):.4f}")
            print(f"Train F1-Score: {train_metrics.get('train/f1_score', 'N/A'):.4f}")
            
            wandb.log(train_metrics)
            
            # Validation
            if self.val_loader:
                val_metrics = self.validate()
                self.val_losses.append(val_metrics['val/loss'])
                self.val_accuracies.append(val_metrics['val/accuracy'])
                self.val_precisions.append(val_metrics.get('val/precision', 0))
                self.val_recalls.append(val_metrics.get('val/recall', 0))
                self.val_f1_scores.append(val_metrics.get('val/f1_score', 0))
                
                # Detailed validation metrics output
                print(f"Val Loss: {val_metrics['val/loss']:.4f}")
                print(f"Val Accuracy: {val_metrics['val/accuracy']:.4f}")
                print(f"Val Precision: {val_metrics.get('val/precision', 'N/A'):.4f}")
                print(f"Val Recall: {val_metrics.get('val/recall', 'N/A'):.4f}")
                print(f"Val F1-Score: {val_metrics.get('val/f1_score', 'N/A'):.4f}")
                
                wandb.log(val_metrics)
                
                # Early Stopping & Model Saving
                if val_metrics['val/loss'] < best_val_loss:
                    best_val_loss = val_metrics['val/loss']
                    patience_counter = 0
                    if save_path:
                        torch.save(self.model.state_dict(), save_path)
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        print(f"\nEarly stopping triggered after epoch {epoch+1}")
                        break
                self.plot_confusion_matrix
                self.plot_metrics
    
    def plot_metrics(self):
        """Plot training and validation metrics."""
        fig, axs = plt.subplots(2, 2, figsize=(20, 15))
        
        # Loss Plot
        axs[0, 0].plot(self.train_losses, label='Training Loss')
        if self.val_loader:
            axs[0, 0].plot(self.val_losses, label='Validation Loss')
        axs[0, 0].set_title('Loss over epochs')
        axs[0, 0].set_xlabel('Epoch')
        axs[0, 0].set_ylabel('Loss')
        axs[0, 0].legend()
        
        # Accuracy Plot
        axs[0, 1].plot(self.train_accuracies, label='Training Accuracy')
        if self.val_loader:
            axs[0, 1].plot(self.val_accuracies, label='Validation Accuracy')
        axs[0, 1].set_title('Accuracy over epochs')
        axs[0, 1].set_xlabel('Epoch')
        axs[0, 1].set_ylabel('Accuracy')
        axs[0, 1].legend()
        
        # Precision Plot
        axs[1, 0].plot(self.train_precisions, label='Training Precision')
        if self.val_loader:
            axs[1, 0].plot(self.val_precisions, label='Validation Precision')
        axs[1, 0].set_title('Precision over epochs')
        axs[1, 0].set_xlabel('Epoch')
        axs[1, 0].set_ylabel('Precision')
        axs[1, 0].legend()
        
        # F1-Score Plot
        axs[1, 1].plot(self.train_f1_scores, label='Training F1-Score')
        if self.val_loader:
            axs[1, 1].plot(self.val_f1_scores, label='Validation F1-Score')
        axs[1, 1].set_title('F1-Score over epochs')
        axs[1, 1].set_xlabel('Epoch')
        axs[1, 1].set_ylabel('F1-Score')
        axs[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig("exploration/outputs/training_metrics.png")
        plt.show()
    
    def plot_confusion_matrix(self, metrics):
        """
        Plot the confusion matrix from validation or training metrics.
        
        Args:
            metrics: Dictionary containing 'confusion_matrix' key
        """
        plt.figure(figsize=(20,20))
        sns.heatmap(
            metrics['confusion_matrix'], 
            annot=True, 
            fmt='d', 
            cmap='Blues'
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('exploration/outputs/confusion_matrix.png')
        
    # TODO das hier muss noch angepasst werden
    def plot_confusion_matrix_normalized(self, metrics):
        
        """
        Plot a normalized confusion matrix from validation or training metrics.
        
        Args:
            metrics: Dictionary containing 'confusion_matrix' key.
        """
        # Normiere die Konfusionsmatrix (Zeilenweise)
        cm = metrics['confusion_matrix']
        cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)

        # Erstelle das Heatmap-Plot
        plt.figure(figsize=(20, 20))
        sns.heatmap(
            cm_normalized, 
            annot=True, 
            fmt='.2f',  # Zwei Dezimalstellen für Prozentsätze
            cmap='Blues',
           
        )
        plt.title('Normalized Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('exploration/outputs/confusion_matrix.png')
        plt.show()
