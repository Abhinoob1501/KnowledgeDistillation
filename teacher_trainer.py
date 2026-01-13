import torch
from teacher import TeacherModel
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from config import Config
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F

class TeacherTrainer:
    def __init__(self):
        
        self.config = Config.teacher_config

    def _validate_teacher(self, teacher, val_loader, device='cuda', use_cached=False):
        """Run validation for a teacher and return val_auc"""
        teacher.eval()
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                if use_cached:
                    features, y = batch
                    features = features.to(device)
                    y = y.to(device)
                    logits = teacher.classifier(features)
                else:
                    x = batch['image'].to(device)
                    y = batch['label'].to(device)
                    logits = teacher(x)
                
                probs = F.softmax(logits, dim=1)[:, 1]
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
        
        try:
            val_auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            val_auc = 0.0
        
        return val_auc

    def train_teachers(self, backbone, grouped_datasets, original_dataset, device='cuda', val_dataset=None, 
                      cached_features=None, cached_val_features=None, class_weights=None):
        
        # Freeze backbone for teacher training
        backbone.freeze_backbone()
        
        # Get num_epochs from config
        num_epochs = Config.teacher_config['num_epochs']
        
        # Unpack cached features if provided
        use_cached = cached_features is not None
        if use_cached:
            train_features, train_labels, train_groups = cached_features
            print("Using cached backbone features for training")
        
        teachers = {}
        for group_id, group_indices in grouped_datasets.items():
            print(f"\n🧑‍🏫 Training teacher for group {group_id}...")

            teacher = TeacherModel(group_id, backbone)
            
            # Freeze backbone parameters
            for param in teacher.backbone.parameters():
                param.requires_grad = False
            
            batch_size = Config.teacher_config.get('batch_size', 150)
            
            # Create dataloader based on whether we use cached features
            if use_cached:
                # Use cached features - create TensorDataset
                group_mask = (train_groups == group_id)
                group_features = train_features[group_mask]
                group_labels = train_labels[group_mask]
                
                from torch.utils.data import TensorDataset
                group_dataset = TensorDataset(group_features, group_labels)
                group_loader = DataLoader(group_dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(42))
            else:
                # Original approach with full dataset
                group_subset = torch.utils.data.Subset(original_dataset, group_indices)
                num_workers = Config.data_config.get('num_workers', 0)
                pin_memory = Config.data_config.get('pin_memory', True)
                prefetch_factor = Config.data_config.get('prefetch_factor', 4) if num_workers > 0 else None
                persistent_workers = Config.data_config.get('persistent_workers', True) and num_workers > 0
                group_loader = DataLoader(group_subset, batch_size=batch_size, shuffle=True,
                                        num_workers=num_workers, pin_memory=pin_memory,
                                        prefetch_factor=prefetch_factor, persistent_workers=persistent_workers,
                                        generator=torch.Generator().manual_seed(42))
            
            lr_value = self.config['optimizer_lr']
            weight_decay_value = self.config['optimizer_weight_decay']
            momentum_value = self.config['momentum']

            if class_weights is not None:
                print(f"⚖️  Using Class Weights passed from pipeline: {class_weights.cpu().numpy()}")

            optimizer = optim.SGD(teacher.classifier.parameters(), lr=lr_value, momentum=momentum_value, weight_decay=weight_decay_value)
            criterion = nn.CrossEntropyLoss(weight=class_weights)

            teacher.to(device)

            # torch.compile optimization (PyTorch 2.0+)
            if Config.compile_config['enabled'] and hasattr(torch, 'compile'):
                print(f"🚀 Compiling teacher model (Group {group_id}) with mode='{Config.compile_config['mode']}'...")
                teacher = torch.compile(
                    teacher,
                    mode=Config.compile_config['mode'],
                    fullgraph=Config.compile_config['fullgraph'],
                    dynamic=Config.compile_config['dynamic']
                )

            # AMP setup
            use_amp = Config.amp_config['enabled'] and device == 'cuda'
            scaler = GradScaler(enabled=use_amp)
            
            # Create validation loader for this group if val_dataset provided
            val_loader = None
            if cached_val_features is not None:
                val_features, val_labels, val_groups = cached_val_features
                val_group_mask = (val_groups == group_id)
                if val_group_mask.any():
                    val_group_features = val_features[val_group_mask]
                    val_group_labels = val_labels[val_group_mask]
                    from torch.utils.data import TensorDataset
                    val_group_dataset = TensorDataset(val_group_features, val_group_labels)
                    val_loader = DataLoader(val_group_dataset, batch_size=batch_size, shuffle=False)
            elif val_dataset is not None and not use_cached:
                val_group_indices = [i for i, item in enumerate(val_dataset) if item['group'].item() == group_id]
                if val_group_indices:
                    val_group_subset = torch.utils.data.Subset(val_dataset, val_group_indices)
                    num_workers = Config.data_config.get('num_workers', 0)
                    pin_memory = Config.data_config.get('pin_memory', True)
                    prefetch_factor = Config.data_config.get('prefetch_factor', 4) if num_workers > 0 else None
                    persistent_workers = Config.data_config.get('persistent_workers', True) and num_workers > 0
                    val_loader = DataLoader(val_group_subset, batch_size=batch_size, shuffle=False,
                                          num_workers=num_workers, pin_memory=pin_memory,
                                          prefetch_factor=prefetch_factor, persistent_workers=persistent_workers)
            
            
            # Early stopping variables
            best_val_auc = 0.0
            patience = Config.teacher_config.get('early_stopping_patience', 5)
            patience_counter = 0
            best_model_state = None

            for epoch in range(num_epochs):
                teacher.train()
                epoch_loss = 0.0
                all_train_probs, all_train_labels = [], []

                pbar = tqdm(group_loader, desc=f"[Group {group_id} | Epoch {epoch+1}/{num_epochs}]")
                for batch in pbar:
                    if use_cached:
                        # Batch is (features, labels) from TensorDataset
                        features, targets = batch
                        features = features.to(device)
                        targets = targets.to(device)
                    else:
                        # Batch is dict from original dataset
                        data = batch['image'].to(device)
                        targets = batch['label'].to(device)

                    optimizer.zero_grad()
                    
                    with autocast(enabled=use_amp):
                        if use_cached:
                            outputs = teacher.classifier(features)
                        else:
                            outputs = teacher(data)
                        loss = criterion(outputs, targets)
                    
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(teacher.classifier.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()

                    epoch_loss += loss.item()
                    avg_loss = epoch_loss / (pbar.n + 1)
                    pbar.set_postfix(loss=avg_loss)
                    
                    # Collect train probs
                    with torch.no_grad():
                        probs = F.softmax(outputs, dim=1)[:, 1]
                        all_train_probs.extend(probs.cpu().numpy())
                        all_train_labels.extend(targets.cpu().numpy())

                avg_train_loss = epoch_loss / len(group_loader)
                
                # Calculate train AUC
                try:
                    train_auc = roc_auc_score(all_train_labels, all_train_probs)
                except ValueError:
                    train_auc = 0.0

                # Run validation
                if val_loader is not None:
                    val_auc = self._validate_teacher(teacher, val_loader, device, use_cached=(cached_val_features is not None))
                    print(f"✅ [Group {group_id}] Epoch {epoch + 1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f}")
                    
                    # Early stopping check
                    if val_auc > best_val_auc:
                        best_val_auc = val_auc
                        patience_counter = 0
                        best_model_state = {k: v.cpu().clone() for k, v in teacher.state_dict().items()}
                        print(f"📈 New best val AUC: {best_val_auc:.4f}")
                    else:
                        patience_counter += 1
                        print(f"⏳ Patience: {patience_counter}/{patience}")
                        
                    if patience_counter >= patience:
                        print(f"🛑 Early stopping triggered after {epoch + 1} epochs")
                        if best_model_state is not None:
                            teacher.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
                            print(f"✅ Restored best model with val AUC: {best_val_auc:.4f}")
                        break
                else:
                    print(f"✅ [Group {group_id}] Epoch {epoch + 1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Train AUC: {train_auc:.4f}")

            teachers[group_id] = teacher

        return teachers
