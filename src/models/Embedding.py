import torch
import torch.nn as nn
import math
from typing import Optional

class Embedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        max_len: int,
        dropout_rate: float = 0.1,
        padding_idx:  int = 0,
    ):
        """
        Args:
            vocab_size: Größe des Vokabulars
            embedding_dim: Dimensionalität der Embeddings
            max_len: Maximale Sequenzlänge
            dropout_rate: Dropout-Rate
            padding_idx: Index für Padding-Token (optional)
        """
        super(Embedding, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.max_len = max_len
        
        # Embedding Layer mit optionalem Padding Index
        # TODO Brauche ich hier eine andere vocab_size? muss das padding token hier mit eingeschlossen werden?
        # Anscheinend brauche ich den, damit das Modell weiß welcher der Padding Index ist und diesen nicht mit berechnet
        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )
        
        # Dropout Layer
        self.dropout = nn.Dropout(p=dropout_rate)
        self.positional_encoding = self._create_positional_encoding(max_len, embedding_dim)
        
        
        # create positional encoding
        self.positional_encoding = self._create_positional_encoding(max_len, embedding_dim)
        
        
    def _create_positional_encoding(self, max_len: int, embedding_dim: int) -> torch.Tensor:
        """
        Erstellt das Positional Encoding Matrix.
        
        Args:
            max_len: Maximale Sequenzlänge
            embedding_dim: Dimensionalität der Embeddings
            
        Returns:
            Positional Encoding Matrix der Form (1, max_len, embedding_dim)
        """
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim)
        )
        
        pos_encoding = torch.zeros(max_len, embedding_dim)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        
        # TODO Warum?
        # Ermöglicht einfache Broadcast-Operationen mit Batch von Embeddings
        # Wandelt die Form von (max_len, embedding_dim) zu (1, max_len, embedding_dim) 
        return pos_encoding.unsqueeze(0)
    
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward Pass des Embedding Layers.
        
        Args:
            x: Input Tensor der Form (batch_size, seq_len)
            attention_mask: Optional mask für die Attention (batch_size, seq_len)
            
        Returns:
            Embedded Tensor der Form (batch_size, seq_len, embedding_dim)
        """
                   
        # Token Embeddings
        embeddings = self.token_embedding(x) * math.sqrt(self.embedding_dim)

        # Positional Encoding hinzufügen
        seq_len = x.size(1)
        embeddings = embeddings + self.positional_encoding[:, :seq_len].to("cuda" if torch.cuda.is_available() else "cpu")
        
        # Wenn Attention Mask vorhanden, maskierte Positionen auf 0 setzen
        # sorgt dafür, dass die Embeddings des Padding Tokens nicht berücksichtigt werden
        embeddings = embeddings * attention_mask.unsqueeze(-1)
            
        # Dropout anwenden
        embeddings = self.dropout(embeddings)
        
        
        return embeddings
