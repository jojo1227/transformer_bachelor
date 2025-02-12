import wandb
from src.models.TransformerEncoder import EncoderClassifier
import yaml
import torch 
from src.datasets.CustomDataset import CustomDataset
import torch.nn as nn
from tqdm import tqdm
from typing import Dict, List, Optional
from transformers import get_linear_schedule_with_warmup


class SweepTrainer:
    
    def __init__(self, num_epochs: int) -> None:
        
        wandb.login()
        self.sweep_id = self.get_sweep_id()
        self.num_epochs = num_epochs
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

            
        
    def start(self):
        wandb.agent(self.sweep_id, function=self.train, count=50)
        

    def get_sweep_id(self):
        
        sweep_configuration = {
            'method': 'bayes',  # Andere Optionen: 'grid', 'bayes'
            'metric': {
                'goal': 'minimize',  # Sie können 'minimize' oder 'maximize' wählen
                'name': 'val/loss'   # Die Metrik, die Sie optimieren möchten
            },
            'parameters': {
                'learning_rate': {
                    'min': 0.00001,
                    'max': 0.0001
                },
                'dropout_rate': {
                    'values': [0.1, 0.2, 0.3, 0.4]
                },
                'batch_size': {
                    'values': [16, 32, 64, 128]
                },
                'optimizer': {
                    'values': ['adam', 'adamw']
                },
                'embedding_dimension': {
                    'values': [16 , 32, 64, 128]
                },
                'num_encoder_layers': {
                    'values': [1, 2, 4, 5]
                },
                'num_heads': {
                    'values': [ 2, 4, 8]
                },
                'criterion': {
                    'values': ['cross_entropy_loss']
                },
                'token_embedding_scaling': {
                    'values': [0, 1]
                },
                'pos_encoding_scaling': {
                    'min': 0.1,
                    'max': 1.0   
                }
                
                    
            
            }
        }
        
        sweep_id = wandb.sweep(sweep=sweep_configuration, project="my-eleventh-sweep")
        return sweep_id



    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train the model for one epoch and collect detailed metrics."""
        self.model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
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
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{correct_predictions/total_predictions:.4f}"
            })
        
        # Compute average metrics
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct_predictions / total_predictions
        
        
        return {
            'train/epoch': epoch,
            'train/loss': avg_loss, 
            'train/accuracy': accuracy,
        }

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate the model on validation dataset."""
        
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
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
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{correct_predictions/total_predictions:.4f}"
            })
            
        
        # Compute average metrics
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct_predictions / total_predictions
        
        
        return {
            'val/epoch': epoch,
            'val/loss': avg_loss, 
            'val/accuracy': accuracy,
        }
        
    def train(self):
        
        wandb.init()

        
        with open('parameter.yaml', 'r') as file:
            static_config = yaml.safe_load(file)
        
        self.model = EncoderClassifier(
            vocab_size=static_config["data"]["vocab_size"],
            num_classes=static_config["data"]["num_classes"],
            embedding_dim=wandb.config.embedding_dimension,
            num_encoder_layers=wandb.config.num_encoder_layers,
            num_heads=wandb.config.num_heads,
            max_len=static_config["data"]["max_seq_length"],
            padding_idx=0,
            dropout_rate=wandb.config.dropout_rate,
            pos_encoding_scaling=wandb.config.pos_encoding_scaling,
            token_embedding_scaling=wandb.config.token_embedding_scaling
        )
        self.model = self.model.to(self.device)
        
        training_data  = CustomDataset("data/encoded/train_sequences.npy", "data/encoded/train_targets.npy")
        self.train_loader = torch.utils.data.DataLoader(training_data, batch_size=wandb.config.batch_size, shuffle=True)
        val_data  = CustomDataset("data/encoded/val_sequences.npy", "data/encoded/val_targets.npy")
        self.val_loader = torch.utils.data.DataLoader(val_data, batch_size=wandb.config.batch_size, shuffle=True)
        
        if(wandb.config.optimizer == 'adam'):
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=wandb.config.learning_rate)
        else: 
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=wandb.config.learning_rate)

        if(wandb.config.criterion == 'cross_entropy_loss'):
            self.criterion = nn.CrossEntropyLoss()
            
    
        
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            
            # Training
            train_metrics = self.train_epoch(epoch)
            print(train_metrics)
            wandb.log({"train/loss": train_metrics["train/loss"]})
            wandb.log({"train/accuracy": train_metrics["train/accuracy"]})
            
            # Validation
            val_metrics = self.validate(epoch)
            wandb.log({"val/loss": val_metrics["val/loss"]})
            wandb.log({"val/accuracy": val_metrics["val/accuracy"]})
            print(val_metrics)
            
        
        
