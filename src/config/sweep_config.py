
sweep_configuration = {
    'method': 'bayes',  # Andere Optionen: 'grid', 'bayes'
    'metric': {
        'goal': 'minimize',  # Sie können 'minimize' oder 'maximize' wählen
        'name': 'val/f1_score'   # Die Metrik, die Sie optimieren möchten
    },
    'parameters': {
        'learning_rate': {
            'min': 0.000001,
            'max': 0.0001
        },
        'dropout_rate': {
            'values': [0.1, 0.2, 0.3, 0.4]
        },
        'batch_size': {
            'values': [ 64, 128, 256]
        },
        'optimizer': {
            'values': ['adam', 'adamw']
        },
        'model_dim': {
            'values': [256, 512]
        },
        'num_encoder_layers': {
            'values': [1, 2, 4, 5, 6]
        },
        'num_heads': {
            'values': [ 2, 4, 8, 16]
        },
        'criterion': {
            'values': ['cross_entropy_loss']
        },
        'pos_encoding_scaling': {
            'values': [0.0, 1.0] 
        }, 
        'activation': {
            'values': ['relu', 'gelu']
        },
        'label_smoothing': {
            'min': 0.0,
            'max': 0.3
        },
        'pooling_type' : {
            'values': ['mean', 'cls']
        },
        'norm_first': {
            'values': [True, False]
        },
        'l1_lambda' : {
            'values': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0]
        },
        'weight_decay': {
            'values': [1e-5, 1e-4, 1e-3, 1e-2]
        },
        'lr_scheduler': {
            'values': ['no', 'CosineAnnealingLR', 'ReduceLROnPlateau']
        },
        'embedding': {
            'values': [True, False]
        },
        'loss_type': {
            'values': ['focal', 'cross_entropy']
        },
        'focal_gamma': {
            'values': [1.0, 2.0, 3.0, 4.0]
        },
        'loss_reduction': {
            'values': ['mean', 'sum']
        },
        'ignore_index': {
            'values': [0]
        }
    }
}
