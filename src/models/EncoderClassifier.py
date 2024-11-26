import torch
import torch.nn as nn
from src.models.Embedding import Embedding


import torch
import torch.nn as nn
from typing import Optional
from .Embedding import Embedding

class EncoderClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embedding_dim: int = 256,
        num_encoder_layers: int = 6,
        num_heads: int = 8,
        max_len: int = 512,
        dropout_rate: float = 0.1,
        padding_idx: int = 0,
    ):
        """
        Args:
            vocab_size: Größe des Vokabulars
            num_classes: Anzahl der Klassifikationsklassen
            embedding_dim: Dimensionalität der Embeddings
            num_encoder_layers: Anzahl der Transformer Encoder Layer
            num_heads: Anzahl der Attention Heads
            max_len: Maximale Sequenzlänge
            dropout_rate: Dropout-Rate
            padding_idx: Index für Padding-Token
        """
        super(EncoderClassifier, self).__init__()
        
        # Embedding Layer
        self.embedding = Embedding(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            max_len=max_len,
            dropout_rate=dropout_rate,
            padding_idx=padding_idx
        )
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            # dim_feedforward=4 * embedding_dim, #default ist hier 2048 
            # dropout=dropout_rate,
            # activation='gelu', #standard ist relu, gelu soll aber angeblich besser sein
            batch_first=True
        )
        
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
            # norm=nn.LayerNorm(embedding_dim)
        )
        
        # Dropout für Regularisierung
        self.dropout = nn.Dropout(dropout_rate)
        
        # Klassifikations-Layer
        # TODO Wie muss ich den hier genau gestalten. 
        # ist das zu kompliziert? 
        # self.classifier = nn.Sequential(
        #    nn.Linear(embedding_dim, embedding_dim),
        #    nn.GELU(),
        #    nn.Dropout(dropout_rate),
        #    nn.Linear(embedding_dim, num_classes)
        #)
        
        self.classifier = nn.Linear(embedding_dim, num_classes)
        
        # Parameter Initialisierung
        # TODO macht das hier einen Unterschied mit dem init weigths 
        # self._init_weights()
        
    def _init_weights(self):
        """Initialisiert die Gewichte des Models"""
        for name, param in self.named_parameters():
            if 'weight' in name and 'norm' not in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward Pass des Models.
        
        Args:
            x: Input Tensor der Form (batch_size, seq_len)
            attention_mask: Attention Mask der Form (batch_size, seq_len)
            
        Returns:
            Logits für jede Klasse (batch_size, num_classes)
        """
        # Embedding Layer
        x = self.embedding(x, attention_mask)
        
        # Transformer Encoder
        if attention_mask is not None:
            # Maske für den Transformer (True für padding positions)
            padding_mask = ~attention_mask.bool()
            x = self.encoder(x, src_key_padding_mask=padding_mask)
        else:
            x = self.encoder(x)
            
        # 
        # Global Max Pooling
        x = self.dropout(x)
        x = x.masked_fill(~attention_mask.unsqueeze(-1), float('-inf')) if attention_mask is not None else x
        # TODO also try class token
        # TODO take mean instead of max, only take mean over entries where attention mask is 1
        x = torch.sum(x * attention_mask, dim=1) / (torch.sum(attention_mask, dim=1) + 1e-8)
        
        # Klassifikation
        logits = self.classifier(x)
        
        return logits
    
    def get_attention_weights(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> list:
        """
        Extrahiert die Attention Weights für Visualisierung.
        Muss im eval() Modus aufgerufen werden.
        
        Returns:
            Liste von Attention Weights für jeden Layer
        """
        attention_weights = []
        
        def hook_fn(module, input, output):
            attention_weights.append(output[1])
            
        hooks = []
        for layer in self.encoder.layers:
            hooks.append(layer.self_attn.register_forward_hook(hook_fn))
            
        # Forward pass
        with torch.no_grad():
            self.forward(x, attention_mask)
            
        # Hooks entfernen
        for hook in hooks:
            hook.remove()
            
        return attention_weights