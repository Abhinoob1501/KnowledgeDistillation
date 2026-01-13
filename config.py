class Config:
    # Data path configuration
    data_path = '/teamspace/studios/this_studio/'
    
    # Dataset configuration
    # Change 'name' to 'OL3I', 'FITZPATRICK17K', or 'PAPILA' to use the respective dataset
    dataset_config = {
        'name': 'PAPILA', # Options: 'HAM10000', 'OL3I', 'FITZPATRICK17K', 'PAPILA'
        'num_classes': 2,
        # Which sensitive attribute to use for fairness/grouping: 'Sex_binary' or 'Age_binary' or 'Skin_type_binary'
        'sensitive_attr': 'Sex_binary',
        
        # Class weighting for imbalanced datasets
        'use_class_weights': True,
        # Weights for OL3I: [0.0436, 0.9564] (Negative, Positive)
        # For FITZPATRICK17K/PAPILA, set to None to auto-calculate
        'class_weights': None 
    }
    
    # Data loading configuration
    data_config = {
        'batch_size': 256,
        'num_workers': 8,
        'pin_memory': True,
        'prefetch_factor': 8,
        'persistent_workers': True
    }
    
    # Feature caching configuration
    disable_cache = False

    # AMP (Automatic Mixed Precision) configuration
    amp_config = {
        'enabled': True,
        'dtype': 'float16'  # 'float16' or 'bfloat16'
    }

    # torch.compile configuration (PyTorch 2.0+)
    compile_config = {
        'enabled': True,
        'mode': 'reduce-overhead',  # 'default', 'reduce-overhead', 'max-autotune'
        'fullgraph': False,
        'dynamic': False
    }

    backbone_trainer_config = {
        'num_epochs': 5,
        'optimizer_lr': 1e-4,
        'optimizer_weight_decay': 0.1,
        'fis_warmup': 0, #Older Implementation max(1, num_epochs // 10)
        'early_stopping_patience': 5  # Early stopping patience
    }

    teacher_config = {
        'num_epochs': 50,
        'optimizer_lr': 1e-2,
        'optimizer_weight_decay': 0,
        'momentum': 0.9, #As in the paper
        'batch_size': 256,
        'early_stopping_patience': 5  # Early stopping patience
    }

    student_config = {
        'tau': 1.5,
        'lambda_kd': 0.95, #0.75
        'num_epochs': 50,
        'fis_warmup': 0, #for classfication 0 for segmentation not provided how much to be done
        'optimizer_lr': 1e-2,
        'momentum' : 0.9,
        'early_stopping_patience': 5  # Early stopping patience
    }

    fis_params = {
        'eps': 1e-8,
    }

    evaluation_config = {
        'batch_size': 256,
        'num_workers': 8,
        'pin_memory': True,
        'prefetch_factor': 8,
        'persistent_workers': True
    }