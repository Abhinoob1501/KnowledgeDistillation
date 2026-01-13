import torch
from torch.utils.data import DataLoader
from backbone import BackboneModel
from backbone_trainer import BackboneTrainer
from teacher_trainer import TeacherTrainer
from student_trainer import Student
from fairness_metrics import evaluate_fairness, print_evaluation_results
from config import Config



from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import numpy as np


def extract_backbone_features(model, dataset, device='cuda', batch_size=200):
    """Extract and cache backbone features for entire dataset"""
    print("Extracting backbone features...")
    model.eval()
    all_features = []
    all_labels = []
    all_groups = []
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting features"):
            data = batch['image'].to(device)
            features = model.get_features(data)
            all_features.append(features.cpu())
            all_labels.append(batch['label'])
            all_groups.append(batch['group'])
    
    features_tensor = torch.cat(all_features, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)
    groups_tensor = torch.cat(all_groups, dim=0)
    
    if len(features_tensor) != len(dataset):
        raise RuntimeError(f"Feature extraction mismatch! Dataset size: {len(dataset)}, Extracted: {len(features_tensor)}")
    
    print(f"Extracted features shape: {features_tensor.shape}")
    return features_tensor, labels_tensor, groups_tensor


def cache_teacher_logits(teachers, features, labels, groups, device='cuda', batch_size=200):
    """Cache teacher logits for each group"""
    print("Caching teacher logits...")
    num_samples = len(features)
    num_classes = len(teachers[list(teachers.keys())[0]].classifier.weight)
    teacher_logits = torch.zeros(num_samples, num_classes)
    
    # Process by group
    for gid, teacher in teachers.items():
        teacher.eval()
        teacher.to(device)
        group_mask = (groups == gid)
        group_indices = torch.where(group_mask)[0]
        
        if len(group_indices) == 0:
            continue
        
        print(f"Caching logits for group {gid} ({len(group_indices)} samples)...")
        
        # Process in batches
        for i in tqdm(range(0, len(group_indices), batch_size), desc=f"Group {gid}"):
            batch_indices = group_indices[i:i+batch_size]
            batch_features = features[batch_indices].to(device)
            
            with torch.no_grad():
                logits = teacher.classifier(batch_features)
                teacher_logits[batch_indices] = logits.cpu()
    
    print(f"Cached teacher logits shape: {teacher_logits.shape}")
    return teacher_logits


def train_fairdi_complete(train_dataset, test_dataset,
                         num_classes=2, device='cuda', backbone='vit_base_patch16_224', val_dataset=None):
    print("Preparing grouped datasets...")
    if hasattr(train_dataset, 'get_groups'):
        groups = train_dataset.get_groups()
    else:
        print("Warning: Dataset does not implement get_groups(). Iterating dataset to find groups (this may be slow)...")
        groups = {}
        for idx in range(len(train_dataset)):
            try:
                item = train_dataset[idx]
                gid = item['group'].item() if torch.is_tensor(item['group']) else item['group']
                groups.setdefault(gid, []).append(idx)
            except Exception as e:
                print(f"Error getting group for index {idx}: {e}")
                continue
        
        print(f'Found {len(groups)} demographic groups')
        for gid, indices in groups.items():
            print(f'Group {gid}: {len(indices)} samples')
    
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
    
    # Create validation loader if val_dataset is provided
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                               pin_memory=pin_memory, prefetch_factor=prefetch_factor,
                               persistent_workers=persistent_workers)

    # --- Centralized Class Weight Calculation ---
    class_weights = None
    if Config.dataset_config.get('use_class_weights', False):
        manual_weights = Config.dataset_config.get('class_weights')
        if manual_weights is not None:
            print(f"Using manual class weights from config: {manual_weights}")
            class_weights = torch.tensor(manual_weights).float().to(device)
        else:
            print("Calculating class weights from training data...")
            # Calculate weights based on inverse frequency
            # Assuming binary classification for now or getting labels from dataset
            if hasattr(train_dataset, 'labels'):
                labels = train_dataset.labels
            elif hasattr(train_dataset, 'targets'):
                labels = train_dataset.targets
            elif hasattr(train_dataset, 'df'):
                # Support for OL3I and HAM10000 which use pandas df
                if 'label_1y' in train_dataset.df.columns:
                    labels = train_dataset.df['label_1y'].values
                elif 'binaryLabel' in train_dataset.df.columns:
                    labels = train_dataset.df['binaryLabel'].values
                else:
                    print("Warning: Could not find 'label_1y' or 'binaryLabel' in dataset.df")
                    labels = None
            
            if labels is None:
                # Fallback: iterate loader (slow)
                print("Warning: Could not access labels directly. Iterating loader to count classes...")
                labels = []
                for batch in train_loader:
                    if isinstance(batch, dict):
                        y = batch['label']
                    else:
                        _, y, _ = batch
                    labels.extend(y.numpy())
                labels = np.array(labels)
            
            class_counts = np.bincount(labels)
            total_samples = len(labels)
            n_classes = len(class_counts)
            weights = total_samples / (n_classes * class_counts)
            
            # Normalize weights to sum to 1 (User preference)
            weights = weights / np.sum(weights)
            
            class_weights = torch.tensor(weights).float().to(device)
            print(f"Calculated class weights (normalized): {class_weights}")
    # --------------------------------------------

    print("\nStep 0: Training backbone with Fair Identity Scaling...")
    backbone_model = BackboneModel(num_classes=num_classes, backbone=backbone).to(device)
    trained_backbone = BackboneTrainer().train(backbone_model, train_loader, groups, device=device, val_loader=val_loader, class_weights=class_weights)

    train_features, train_labels, train_groups = None, None, None
    val_features, val_labels, val_groups = None, None, None
    train_teacher_logits = None
    val_teacher_logits = None

    if not getattr(Config, 'disable_cache', False):
        print("\nStep 0.5: Extracting and caching backbone features...")
        
        # Temporarily switch to deterministic transforms for training set
        original_transform = None
        if hasattr(train_dataset, 'transform'):
            original_transform = train_dataset.transform
            if val_dataset is not None and hasattr(val_dataset, 'transform'):
                print("Switching train_dataset to deterministic transforms (from val_dataset) for feature extraction...")
                train_dataset.transform = val_dataset.transform
            else:
                # Try importing from ham10000 or ol3i as fallback
                dataset_name = Config.dataset_config.get('name', 'HAM10000')
                try:
                    if dataset_name == 'OL3I':
                        from ol3i import get_ol3i_transforms
                        print("Switching train_dataset to deterministic transforms (using ol3i.get_ol3i_transforms)...")
                        train_dataset.transform = get_ol3i_transforms(is_training=False)
                    else:
                        from ham10000 import get_transforms
                        print("Switching train_dataset to deterministic transforms (using ham10000.get_transforms)...")
                        train_dataset.transform = get_transforms(is_training=False)
                except ImportError:
                    print("Warning: Could not switch to deterministic transforms. Features might be extracted with random augmentations.")

        train_features, train_labels, train_groups = extract_backbone_features(trained_backbone, train_dataset, device, batch_size)

        # Restore transform
        if original_transform is not None:
            train_dataset.transform = original_transform
            print("Restored original training transforms.")

        if val_dataset is not None:
            val_features, val_labels, val_groups = extract_backbone_features(trained_backbone, val_dataset, device, batch_size)
    else:
        print("\n⚠️ Feature caching disabled by config. Training will use on-the-fly augmentation.")

    print("\nStep 1: Training specialized teachers...")
    teachers = TeacherTrainer().train_teachers(trained_backbone, groups, train_dataset, device=device, 
                                               val_dataset=val_dataset, 
                                               cached_features=(train_features, train_labels, train_groups) if train_features is not None else None,
                                               cached_val_features=(val_features, val_labels, val_groups) if val_features is not None else None,
                                               class_weights=class_weights)

    if not getattr(Config, 'disable_cache', False):
        print("\nStep 1.5: Caching teacher logits...")
        train_teacher_logits = cache_teacher_logits(teachers, train_features, train_labels, train_groups, device, batch_size)
        if val_dataset is not None:
            val_teacher_logits = cache_teacher_logits(teachers, val_features, val_labels, val_groups, device, batch_size)

    print("\nStep 2: Training student with knowledge distillation...")
    student_model = Student(trained_backbone).to(device)
    
    # Prepare cached validation features for student if available
    cached_val_for_student = None
    if val_dataset is not None and val_features is not None:
        cached_val_for_student = (val_features, val_labels, val_groups, val_teacher_logits)
        print(f"✅ Validation enabled: {len(val_labels)} validation samples")
    elif val_dataset is not None:
         print(f"✅ Validation enabled (uncached): {len(val_dataset)} validation samples")
    else:
        print("⚠️  WARNING: No validation dataset provided - early stopping will NOT work for student!")
    
    cached_features_for_student = None
    if train_features is not None:
        cached_features_for_student = (train_features, train_labels, train_groups, train_teacher_logits)

    student = student_model.train_student(teachers, train_loader, device=device, val_loader=val_loader,
                                         cached_features=cached_features_for_student,
                                         cached_val_features=cached_val_for_student, groups=groups,
                                         class_weights=class_weights)

    print("\nEvaluating final model...")
    results = evaluate_fairness(student, test_loader, device)
    print_evaluation_results(results)

    return teachers, student, results
