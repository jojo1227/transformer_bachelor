# config/config.py

class ModelConfig:
    def __init__(self):
        # Daten-Parameter
        self.data_directory = '/home/johannes/cadets_jannik'
        self.unusable_threshold = 10
        self.rare_threshold = 50
        self.splits = (0.7, 0.15, 0.15)
        self.max_sequence_length = 100

        # Model-Parameter
        self.model_dim = 512
        self.num_encoder_layers = 6
        self.num_heads = 8
        self.dim_feed_forward = self.model_dim * 4
        self.dropout_rate = 0.1
        self.activation = 'gelu'
        self.padding_idx = 0
        self.pos_encoding_scaling = 1.0
        self.pooling_type = 'mean'
        self.norm_first = True
        self.embedding = True

        # Training-Parameter
        self.num_epochs = 200
        self.early_stopping_patience = 30
        self.batch_size = 128
        self.learning_rate = 1e-4
        self.weight_decay = 1e-4
        self.l1_lambda = 1e-5
        
        # Loss Function Parameter
        self.loss_type = 'focal'  # 'focal' oder 'cross_entropy'
        self.focal_gamma = 2.0
        self.label_smoothing = 0.1
        self.loss_reduction = 'mean'
        self.ignore_index = 0
        
        # Optimizer und Scheduler
        self.optimizer = 'adamw'
        self.lr_scheduler = 'CosineAnnealingLR'

config = ModelConfig()