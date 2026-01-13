import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from config import Config
from fis_loss import FISLoss

from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
import numpy as np

class BackboneTrainer:
    # def _set_trainable_layers(self, model):
    #     # Unfreeze last two transformer encoder layers
    #     for name, param in model.backbone.named_parameters():
    #         if 'encoder.layer.10' in name or 'encoder.layer.11' in name:
    #             param.requires_grad = True
    #         else:
    #             param.requires_grad = False

    #     # Always unfreeze classifier head
    #     for p in model.classifier.parameters():
    #         p.requires_grad = True

    def _set_trainable_layers(self, model):
        # Define which layers to unfreeze (e.g., last 4 layers)
        # Assuming a 12-layer model, indices are 0-11.
        trainable_layer_indices = {'8', '9', '10', '11'}
        
        for name, param in model.backbone.named_parameters():
            # Check if the parameter belongs to one of the target layers
            # logic: checks if 'encoder.layer.X' matches our list
            is_trainable = False
            
            # 1. Check if it falls in our target encoder layers
            for layer_idx in trainable_layer_indices:
                if f'encoder.layer.{layer_idx}' in name:
                    is_trainable = True
                    break
            
            # 2. Also unfreeze LayerNorms (usually good practice when fine-tuning deep)
            if 'layernorm' in name.lower():
                 is_trainable = True

            param.requires_grad = is_trainable

        # Always unfreeze classifier head
        for p in model.classifier.parameters():
            p.requires_grad = True    
            
    def _validate_worst_case(self, model, val_loader, groups, device='cuda'):
        """Run validation and return worst-case (minimum) AUC across groups"""
        model.eval()
        all_probs = []
        all_labels = []
        all_groups = []
        
        with torch.no_grad():
            for batch in val_loader:
                data = batch['image'].to(device)
                labels = batch['label'].to(device)
                group_ids = batch['group'].to(device)
                
                outputs = model(data)
                probs = F.softmax(outputs, dim=1)[:, 1]
                
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_groups.extend(group_ids.cpu().numpy())
        
        # Calculate group-wise AUCs
        import numpy as np
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
        model.train()
        return worst_case_auc

    def train(self, model, dataloader, groups, config=False, device='cuda', val_loader=None, class_weights=None):
        self.config = Config.backbone_trainer_config if config is False else config
        num_epochs = self.config['num_epochs']

        self._set_trainable_layers(model)

        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.config['optimizer_lr'],
            weight_decay=self.config['optimizer_weight_decay']
        )
        
        if class_weights is not None:
            print(f"⚖️  Using Class Weights passed from pipeline: {class_weights.cpu().numpy()}")

        # Pass class weights to CrossEntropyLoss, then to FISLoss
        base_loss = nn.CrossEntropyLoss(weight=class_weights, reduction='none')
        fis_loss = FISLoss(base_loss_fn=base_loss)
        
        model.to(device)

        # torch.compile optimization (PyTorch 2.0+)
        if Config.compile_config['enabled'] and hasattr(torch, 'compile'):
            print(f"🚀 Compiling backbone model with mode='{Config.compile_config['mode']}'...")
            model = torch.compile(
                model,
                mode=Config.compile_config['mode'],
                fullgraph=Config.compile_config['fullgraph'],
                dynamic=Config.compile_config['dynamic']
            )

        # AMP setup
        use_amp = Config.amp_config['enabled'] and device == 'cuda'
        scaler = GradScaler(enabled=use_amp)
        
        # Early stopping variables
        best_worst_case_auc = 0.0
        patience = self.config.get('early_stopping_patience', 5)
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0
            all_train_probs = []
            all_train_labels = []

            pbar = tqdm(dataloader, desc=f"[Backbone Epoch {epoch + 1}/{num_epochs}]")
            for batch in pbar:
                data = batch['image'].to(device)
                labels = batch['label'].to(device)
                group_ids = batch['group'].to(device)

                self.optimizer.zero_grad()
                
                with autocast(enabled=use_amp):
                    outputs = model(data)
                    loss = fis_loss(outputs, labels, group_ids)
                
                scaler.scale(loss).backward()
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(self.optimizer)
                scaler.update()

                epoch_loss += loss.item()
                avg_loss = epoch_loss / (pbar.n + 1)
                pbar.set_postfix(loss=avg_loss)
                
                # Collect train probs for AUC
                with torch.no_grad():
                    probs = F.softmax(outputs, dim=1)[:, 1]
                    all_train_probs.extend(probs.cpu().numpy())
                    all_train_labels.extend(labels.cpu().numpy())

            avg_train_loss = epoch_loss / len(dataloader)
            
            # Calculate train AUC
            try:
                train_auc = roc_auc_score(all_train_labels, all_train_probs)
            except ValueError:
                train_auc = 0.0
            
            # Run validation every epoch
            if val_loader is not None:
                worst_case_auc = self._validate_worst_case(model, val_loader, groups, device)
                print(f"✅ Epoch {epoch + 1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Train AUC: {train_auc:.4f} | Val Worst-Case AUC: {worst_case_auc:.4f}")
                
                # Early stopping check
                if worst_case_auc > best_worst_case_auc:
                    best_worst_case_auc = worst_case_auc
                    patience_counter = 0
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    print(f"📈 New best worst-case AUC: {best_worst_case_auc:.4f}")
                else:
                    patience_counter += 1
                    print(f"⏳ Patience: {patience_counter}/{patience}")
                    
                if patience_counter >= patience:
                    print(f"🛑 Early stopping triggered after {epoch + 1} epochs")
                    if best_model_state is not None:
                        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
                        print(f"✅ Restored best model with worst-case AUC: {best_worst_case_auc:.4f}")
                    break
            else:
                print(f"✅ Epoch {epoch + 1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Train AUC: {train_auc:.4f}")

        return model