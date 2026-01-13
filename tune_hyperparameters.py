import optuna
import torch
import numpy as np
import os
from config import Config
import main
from backbone import BackboneModel
from backbone_trainer import BackboneTrainer
from teacher_trainer import TeacherTrainer
from student_trainer import Student
from run_full_pipeline import extract_backbone_features, cache_teacher_logits
from fairness_metrics import evaluate_fairness
from torch.utils.data import DataLoader

# Global cache for pipeline components to avoid re-running backbone/teacher training
PIPELINE_CACHE = None

def prepare_pipeline_components():
    """
    Runs the initial stages of the pipeline (Backbone -> Teachers) once.
    Returns all components needed to train the student.
    """
    print("="*50)
    print("PREPARING PIPELINE COMPONENTS (ONE-TIME SETUP)")
    print("="*50)
    
    # 1. Setup Data
    main.set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    dataset_name = Config.dataset_config.get('name', 'HAM10000')
    data_path = Config.data_path
    
    train_dataset, val_dataset, test_dataset = main.get_datasets(dataset_name, data_path)
    
    # Get groups
    print("Extracting groups...")
    if hasattr(train_dataset, 'get_groups'):
        groups = train_dataset.get_groups()
    else:
        groups = {}
        for idx in range(len(train_dataset)):
            try:
                item = train_dataset[idx]
                gid = item['group'].item() if torch.is_tensor(item['group']) else item['group']
                groups.setdefault(gid, []).append(idx)
            except Exception:
                continue
    
    # Setup Loaders
    batch_size = Config.data_config['batch_size']
    num_workers = Config.data_config['num_workers']
    pin_memory = Config.data_config['pin_memory']
    prefetch_factor = Config.data_config['prefetch_factor'] if num_workers > 0 else None
    persistent_workers = Config.data_config['persistent_workers'] and num_workers > 0
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                             pin_memory=pin_memory, prefetch_factor=prefetch_factor,
                             persistent_workers=persistent_workers,
                             generator=torch.Generator().manual_seed(42))
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            pin_memory=pin_memory, prefetch_factor=prefetch_factor,
                            persistent_workers=persistent_workers)
    
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                               pin_memory=pin_memory, prefetch_factor=prefetch_factor,
                               persistent_workers=persistent_workers)

    # 2. Train Backbone
    print("\nStep 0: Training backbone...")
    num_classes = Config.dataset_config.get('num_classes', 2)
    backbone_model = BackboneModel(num_classes=num_classes, backbone='vit_base_patch16_224').to(device)
    trained_backbone = BackboneTrainer().train(backbone_model, train_loader, groups, device=device, val_loader=val_loader)

    # 3. Extract Features
    print("\nStep 0.5: Extracting backbone features...")
    train_features, train_labels, train_groups = extract_backbone_features(trained_backbone, train_dataset, device, batch_size)
    
    val_features, val_labels, val_groups = None, None, None
    if val_dataset is not None:
        val_features, val_labels, val_groups = extract_backbone_features(trained_backbone, val_dataset, device, batch_size)

    # 4. Train Teachers
    print("\nStep 1: Training specialized teachers...")
    teachers = TeacherTrainer().train_teachers(trained_backbone, groups, train_dataset, device=device, 
                                               val_dataset=val_dataset, cached_features=(train_features, train_labels, train_groups),
                                               cached_val_features=(val_features, val_labels, val_groups) if val_dataset else None)

    # 5. Cache Logits
    print("\nStep 1.5: Caching teacher logits...")
    train_teacher_logits = cache_teacher_logits(teachers, train_features, train_labels, train_groups, device, batch_size)
    
    val_teacher_logits = None
    if val_dataset is not None:
        val_teacher_logits = cache_teacher_logits(teachers, val_features, val_labels, val_groups, device, batch_size)

    # Prepare cached tuples
    cached_features = (train_features, train_labels, train_groups, train_teacher_logits)
    cached_val_features = None
    if val_dataset is not None and val_features is not None:
        cached_val_features = (val_features, val_labels, val_groups, val_teacher_logits)

    return {
        'backbone': trained_backbone,
        'teachers': teachers,
        'train_loader': train_loader,
        'test_loader': test_loader,
        'val_loader': val_loader,
        'cached_features': cached_features,
        'cached_val_features': cached_val_features,
        'groups': groups,
        'device': device
    }

def objective(trial):
    global PIPELINE_CACHE
    if PIPELINE_CACHE is None:
        PIPELINE_CACHE = prepare_pipeline_components()
    
    c = PIPELINE_CACHE
    
    # -------------------------------------------------
    # Hyperparameters to tune
    # -------------------------------------------------
    lambda_kd = trial.suggest_float('lambda_kd', 0.0, 1.0)
    tau = trial.suggest_float('tau', 1.0, 3.0)
    
    print(f"\n--- Trial {trial.number} ---")
    print(f"Params: lambda_kd={lambda_kd:.4f}, tau={tau:.4f}")
    
    # Update Config
    Config.student_config['lambda_kd'] = lambda_kd
    Config.student_config['tau'] = tau
    
    # -------------------------------------------------
    # Train Student
    # -------------------------------------------------
    # Create a new student instance for each trial
    student_model = Student(c['backbone']).to(c['device'])
    
    student = student_model.train_student(
        teachers=c['teachers'], 
        full_dataloader=c['train_loader'], 
        device=c['device'], 
        val_loader=c['val_loader'],
        cached_features=c['cached_features'],
        cached_val_features=c['cached_val_features'], 
        groups=c['groups']
    )
    
    # -------------------------------------------------
    # Evaluate
    # -------------------------------------------------
    results = evaluate_fairness(student, c['test_loader'], c['device'])
    
    overall_auc = results['overall_auc']
    auc_gap = results['auc_gap']
    
    # Handle potential missing metrics safely
    fl_metrics = results.get('fairlearn_metrics', {})
    dpd = fl_metrics.get('demographic_parity_difference', 0.0)
    eod = fl_metrics.get('equalized_odds_difference', 0.0)
    
    # Log metrics to Optuna
    trial.set_user_attr("overall_auc", overall_auc)
    trial.set_user_attr("auc_gap", auc_gap)
    trial.set_user_attr("dpd", dpd)
    trial.set_user_attr("eod", eod)
    
    print(f"Results: AUC={overall_auc:.4f}, Gap={auc_gap:.4f}, DPD={dpd:.4f}, EOD={eod:.4f}")
    
    # -------------------------------------------------
    # Combined Objective
    # -------------------------------------------------
    # Maximize Score. 
    # We want High AUC, Low Gap, Low DPD, Low EOD.
    # Weights: High emphasis on fairness metrics.
    
    # Score = AUC - (4.0 * Gap + 1.5 * DPD + 1.5 * EOD)
    score = overall_auc - (4.0 * auc_gap + 1.5 * dpd + 1.5 * eod)
    
    return score

if __name__ == "__main__":
    # Ensure storage directory exists
    db_path = "sqlite:///fairness_tuning.db"
    
    print("Starting Optuna Hyperparameter Tuning...")
    study = optuna.create_study(
        direction="maximize",
        storage=db_path,
        study_name="fairdi_tuning",
        load_if_exists=True
    )
    
    study.optimize(objective, n_trials=20)
    
    print("\n" + "="*50)
    print("TUNING COMPLETE")
    print("="*50)
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value (Score): {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    print("\n  Metrics:")
    for key, value in trial.user_attrs.items():
        print(f"    {key}: {value}")
