# training/trainer.py
import torch
from tqdm import tqdm
import wandb

class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler, config, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.best_f1_score = float('inf')
        self.best_val_score = float('inf')
        self.patience_counter = 0

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for sequences, labels, attention_masks in progress_bar:
            sequences = sequences.to(self.device)
            labels = labels.to(self.device)
            attention_masks = attention_masks.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(sequences, attention_masks)
            
            labels = labels.squeeze(1)
            labels = torch.argmax(labels, dim=1)
            
            loss = self.criterion(outputs, labels)
            
            # L1 Regularisierung
            if self.config.l1_lambda > 0:
                l1_norm = sum(p.abs().sum() for p in self.model.parameters())
                loss = loss + self.config.l1_lambda * l1_norm
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        return self.compute_metrics(all_labels, all_predictions, total_loss, len(train_loader), prefix='train')

    @torch.no_grad()
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        for sequences, labels, attention_masks in tqdm(val_loader, desc="Validating"):
            sequences = sequences.to(self.device)
            labels = labels.to(self.device)
            attention_masks = attention_masks.to(self.device)
            
            outputs = self.model(sequences, attention_masks)
            labels = labels.squeeze(1)
            labels = torch.argmax(labels, dim=1)
            
            loss = self.criterion(outputs, labels)
            total_loss += loss.item()
            
            predictions = torch.argmax(outputs, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        metrics = self.compute_metrics(all_labels, all_predictions, total_loss, len(val_loader), prefix='val')
        self.update_scheduler(metrics['val/loss'])
        
        return metrics

    def train(self, train_loader, val_loader, save_dir, class_names):
        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.config.num_epochs}")
            
            # Log learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            wandb.log({"learning_rate": current_lr}, step=epoch)
            
            # Training
            train_metrics = self.train_epoch(train_loader)
            self.log_metrics(train_metrics, epoch)
            
            # Validation
            val_metrics = self.validate(val_loader)
            self.log_metrics(val_metrics, epoch)
            
            # Speichere Confusion Matrix
            plot_confusion_matrix(
                val_metrics['val/confusion_matrix'],
                epoch,
                save_dir,
                class_names
            )
            
            # Early Stopping & Model Saving
            if self.check_early_stopping(val_metrics['val/f1_score'], epoch, save_dir):
                break