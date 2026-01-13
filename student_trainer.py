import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from config import Config
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from fis_loss import FISLoss
import numpy as np

class Student(nn.Module):
    def __init__(self, backbone, config=False):
        super().__init__()
        self.config = Config.student_config if not config else config
        self.backbone = backbone

        feature_dim = backbone.feature_dim if hasattr(backbone, 'feature_dim') else backbone.fc.in_features
        num_classes = backbone.num_classes if hasattr(backbone, 'num_classes') else 2
        self.classifier = nn.Linear(feature_dim, num_classes)

        self.tau = self.config['tau']
        self.lambda_kd = self.config['lambda_kd']
        self.num_epochs = self.config['num_epochs']

    # ------------------------------------------------------------------
    def forward(self, x):
        features = self.backbone.get_features(x) if hasattr(self.backbone, 'get_features') else self.backbone(x)
        return self.classifier(features)

    # ------------------------------------------------------------------
    def _validate_student_worst_case(self, teachers, val_loader, groups, device='cuda', cached_val_features=None):
        """Run validation and return worst-case (minimum) AUC across groups for student"""
        self.eval()
        all_probs = []
        all_labels = []
        all_groups = []
        
        use_cached = cached_val_features is not None
        
        if use_cached:
            val_features, val_labels, val_groups, val_teacher_logits = cached_val_features
            cached_dataset = TensorDataset(val_features, val_labels, val_groups, val_teacher_logits)
            val_loader = DataLoader(cached_dataset, batch_size=Config.data_config['batch_size'], shuffle=False, generator=torch.Generator().manual_seed(42))
        
        with torch.no_grad():
            for batch in val_loader:
                if use_cached:
                    features, targets, group_ids, teacher_logits_batch = batch
                    features = features.to(device)
                    targets = targets.to(device)
                    student_logits = self.classifier(features).detach().clone()
                else:
                    data = batch['image'].to(device)
                    targets = batch['label'].to(device)
                    group_ids = batch['group'].to(device)
                    student_logits = self.forward(data).detach().clone()
                
                probs = F.softmax(student_logits, dim=1)[:, 1]
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(targets.cpu().numpy())
                all_groups.extend(group_ids.cpu().numpy())
        
        # Calculate group-wise AUCs
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        all_groups = np.array(all_groups)
        
        group_aucs = {}
        for group_id in groups.keys():
            group_mask = (all_groups == group_id)
            if np.sum(group_mask) > 0:
                group_labels = all_labels[group_mask]
                group_probs = all_probs[group_mask]
                if len(np.unique(group_labels)) > 1:
                    try:
                        group_aucs[group_id] = roc_auc_score(group_labels, group_probs)
                    except ValueError:
                        group_aucs[group_id] = 0.0
        
        worst_case_auc = min(group_aucs.values()) if group_aucs else 0.0
        self.train()
        return worst_case_auc
    
    def train_student(self, teachers, full_dataloader, device='cuda', val_loader=None , cached_features = None, cached_val_features=None, groups=None, class_weights=None):
        self.backbone.freeze_backbone()
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Move to device first
        self.to(device)

        # torch.compile optimization (PyTorch 2.0+)
        if Config.compile_config['enabled'] and hasattr(torch, 'compile'):
            print(f"🚀 Compiling student model with mode='{Config.compile_config['mode']}'...")
            # Compile the forward pass
            self.forward = torch.compile(
                self.forward,
                mode=Config.compile_config['mode'],
                fullgraph=Config.compile_config['fullgraph'],
                dynamic=Config.compile_config['dynamic']
            )
        lr_value= self.config['optimizer_lr']
        momentum_value = self.config['momentum']
        self.optimizer = optim.SGD(self.classifier.parameters(), lr=lr_value, momentum=momentum_value)
        kl_loss = nn.KLDivLoss(reduction='batchmean', log_target=True)
        
        # Check if using cached features and logits
        use_cached = cached_features is not None
        
        if use_cached:
            train_features, train_labels, train_groups, train_teacher_logits = cached_features
            print("Using cached features and teacher logits for training")
            # Create dataloader from cached data
            cached_dataset = TensorDataset(train_features, train_labels, train_groups, train_teacher_logits)
            full_dataloader = DataLoader(cached_dataset, batch_size=Config.data_config['batch_size'], shuffle=True, generator=torch.Generator().manual_seed(42))
        
        if class_weights is not None:
            print(f"⚖️  Using Class Weights passed from pipeline: {class_weights.cpu().numpy()}")

        # Pass class weights to CrossEntropyLoss, then to FISLoss
        base_loss = nn.CrossEntropyLoss(weight=class_weights, reduction='none')
        fis_loss = FISLoss(base_loss_fn=base_loss)

        for t in teachers.values():
            t.eval()

        self.classifier.train()
        log_every = 1  # set to 1 for every epoch, or change to 10/20 etc.

        # AMP setup
        use_amp = Config.amp_config['enabled'] and device == 'cuda'
        scaler = GradScaler(enabled=use_amp)
        
        # Early stopping variables
        best_worst_case_auc = 0.0
        patience = self.config.get('early_stopping_patience', 5)
        patience_counter = 0
        best_model_state = None
        
        # Debug: Check validation setup
        if val_loader is None:
            print("⚠️  WARNING: val_loader is None - validation disabled!")
        if groups is None:
            print("⚠️  WARNING: groups is None - validation disabled!")
        if val_loader is not None and groups is not None:
            print(f"✅ Validation enabled with patience={patience}")

        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            all_probs = []
            all_labels = []

            pbar = tqdm(full_dataloader, desc=f"[Student | Epoch {epoch+1}/{self.num_epochs}]")
            for batch in pbar:
                if use_cached:
                    # Batch is (features, labels, groups, teacher_logits) from cached data
                    features, targets, group_ids, teacher_logits_batch = batch
                    features = features.to(device)
                    targets = targets.to(device)
                    teacher_logits_batch = teacher_logits_batch.to(device)
                else:
                    # Original approach
                    data = batch['image'].to(device)
                    targets = batch['label'].to(device)
                    group_ids = batch['group'].to(device)

                self.optimizer.zero_grad()
                
                with autocast(enabled=use_amp):
                    if use_cached:
                        student_logits = self.classifier(features)
                        student_log_probs = F.log_softmax(student_logits / self.tau, dim=1)
                        teacher_log_probs = F.log_softmax(teacher_logits_batch / self.tau, dim=1)
                        # Ensure group_ids is on the correct device for FIS loss
                        group_ids = group_ids.to(device)
                    else:
                        student_logits = self.forward(data)
                        student_log_probs = F.log_softmax(student_logits / self.tau, dim=1)

                        # Batch teacher inference
                        # We pass the FULL batch to the teacher to ensure constant input shape for torch.compile
                        # Then we slice the results for the specific group.
                        teacher_log_probs_dict = {}
                        with torch.no_grad():
                            for gid in teachers.keys():
                                # Find all samples belonging to this group
                                group_mask = (group_ids == gid)
                                if group_mask.any():
                                    teacher = teachers[gid].to(device)
                                    
                                    # Pass FULL data (constant shape) instead of group_data (variable shape)
                                    # This prevents torch.compile recompilation issues
                                    all_t_logits = teacher(data)
                                    
                                    # Slice only the logits we need
                                    t_logits = all_t_logits[group_mask]
                                    teacher_log_probs_dict[gid] = F.log_softmax(t_logits / self.tau, dim=1)
                        
                        # Reconstruct teacher log probs in original order
                        teacher_log_probs = torch.zeros_like(student_log_probs)
                        for gid in teachers.keys():
                            group_mask = (group_ids == gid)
                            if group_mask.any():
                                teacher_log_probs[group_mask] = teacher_log_probs_dict[gid]

                    distill = kl_loss(student_log_probs, teacher_log_probs)
                    supervise = fis_loss(student_logits, targets, group_ids) #fis_loss(outputs, labels, group_ids)
                    total = self.lambda_kd * (self.tau ** 2) * distill + (1 - self.lambda_kd) * supervise

                scaler.scale(total).backward()
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), max_norm=1.0)
                scaler.step(self.optimizer)
                scaler.update()

                epoch_loss += total.item()
                avg_loss = epoch_loss / (pbar.n + 1)
                pbar.set_postfix(loss=avg_loss)

                # collect for AUC
                with torch.no_grad():
                    probs = F.softmax(student_logits, dim=1)[:, 1]
                all_probs.extend(probs.detach().cpu().numpy())
                all_labels.extend(targets.detach().cpu().numpy())

            avg_train_loss = epoch_loss / len(full_dataloader)
            
            # Calculate train AUC
            try:
                train_auc = roc_auc_score(all_labels, all_probs)
            except ValueError:
                train_auc = 0.0

            # Run validation every epoch
            if val_loader is not None and groups is not None:
                worst_case_auc = self._validate_student_worst_case(teachers, val_loader, groups, device, cached_val_features)
                print(f"✅ Epoch {epoch + 1}/{self.num_epochs} | Train Loss: {avg_train_loss:.4f} | Train AUC: {train_auc:.4f} | Val Worst-Case AUC: {worst_case_auc:.4f}")
                
                # Early stopping check
                if worst_case_auc > best_worst_case_auc:
                    best_worst_case_auc = worst_case_auc
                    patience_counter = 0
                    best_model_state = {k: v.cpu().clone() for k, v in self.state_dict().items()}
                    print(f"📈 New best worst-case AUC: {best_worst_case_auc:.4f}")
                else:
                    patience_counter += 1
                    print(f"⏳ Patience: {patience_counter}/{patience}")
                    
                if patience_counter >= patience:
                    print(f"🛑 Early stopping triggered after {epoch + 1} epochs")
                    if best_model_state is not None:
                        self.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
                        print(f"✅ Restored best model with worst-case AUC: {best_worst_case_auc:.4f}")
                    break
            else:
                print(f"✅ Epoch {epoch + 1}/{self.num_epochs} | Train Loss: {avg_train_loss:.4f} | Train AUC: {train_auc:.4f}")

        return self