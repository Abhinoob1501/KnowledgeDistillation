from sklearn.metrics import roc_auc_score
import torch
import numpy as np
from fairlearn.metrics import (
    equalized_odds_difference,
    equalized_odds_ratio,
    demographic_parity_difference,
    demographic_parity_ratio,
)

# Equalized odds Difference
def equiodds_difference(preds, labels, attrs):
    print("Preds: ", np.unique(preds))
    print("Labels: ", np.unique(labels.cpu()))
    print("Attrs: ", np.unique(attrs, return_counts=True))
    print("\n")
    return equalized_odds_difference(labels.cpu(), preds, sensitive_features=attrs)

# Equalized odds Ratio
def equiodds_ratio(preds, labels, attrs):
    return equalized_odds_ratio(labels.cpu(), preds, sensitive_features=attrs)

# Demographic Parity Difference
def dpd(preds, labels, attrs):
    return demographic_parity_difference(labels.cpu(), preds, sensitive_features=attrs)

# Demographic Parity Ratio
def dpr(preds, labels, attrs):
    return demographic_parity_ratio(labels.cpu(), preds, sensitive_features=attrs)

def evaluate_fairness(model, test_loader, device='cuda'):
    model.eval()
    all_predictions = []
    all_probabilities = []
    all_labels = []
    all_groups = []
    
    with torch.no_grad():
        for batch in test_loader:
            data = batch['image'].to(device)
            labels = batch['label'].to(device)
            group_ids = batch['group'].to(device)
            
            outputs = model(data)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[:, 1]  # Positive class probabilities
            predictions = torch.argmax(outputs, dim=1)  # Binary predictions for fairlearn
            
            all_predictions.append(predictions.cpu())
            all_probabilities.append(probabilities.cpu())
            all_labels.append(labels.cpu())
            all_groups.append(group_ids.cpu())
    
    all_predictions = torch.cat(all_predictions)
    all_probabilities = torch.cat(all_probabilities)
    all_labels = torch.cat(all_labels)
    all_groups = torch.cat(all_groups)
    
    # Diagnostic: Check prediction distribution
    unique_preds, counts_preds = torch.unique(all_predictions, return_counts=True)
    pred_dist = dict(zip(unique_preds.tolist(), counts_preds.tolist()))
    print(f"\n[Diagnostics] Prediction Distribution: {pred_dist}")
    if len(unique_preds) == 1:
        print("⚠️  Warning: Model is predicting only one class! This causes NaN in fairness ratios.")
        print("   Possible causes: Class imbalance, poor convergence, or threshold mismatch.")

    # Original AUC calculations
    group_aucs = {}
    unique_groups = torch.unique(all_groups).tolist()
    unique_groups.sort()
    for group_id in unique_groups:
        group_mask = (all_groups == group_id)
        if torch.sum(group_mask) > 0:
            group_labels = all_labels[group_mask]
            group_probs = all_probabilities[group_mask]
            if len(torch.unique(group_labels)) > 1:  # Check for both classes
                group_aucs[group_id] = roc_auc_score(group_labels.numpy(), group_probs.numpy())
    
    try:
        overall_auc = roc_auc_score(all_labels.numpy(), all_probabilities.numpy())
    except ValueError:
        overall_auc = float('nan')
    
    min_auc = min(group_aucs.values()) if group_aucs else 0.0
    max_auc = max(group_aucs.values()) if group_aucs else 0.0
    auc_gap = max_auc - min_auc
    
    # Fairlearn metrics
    fairlearn_metrics = {}
    try:
        fairlearn_metrics['equalized_odds_difference'] = equiodds_difference(
            all_predictions.numpy(), all_labels, all_groups.numpy()
        )
        fairlearn_metrics['equalized_odds_ratio'] = equiodds_ratio(
            all_predictions.numpy(), all_labels, all_groups.numpy()
        )
        fairlearn_metrics['demographic_parity_difference'] = dpd(
            all_predictions.numpy(), all_labels, all_groups.numpy()
        )
        fairlearn_metrics['demographic_parity_ratio'] = dpr(
            all_predictions.numpy(), all_labels, all_groups.numpy()
        )
    except Exception as e:
        print(f"Error calculating fairlearn metrics: {e}")
        fairlearn_metrics = {
            'equalized_odds_difference': 0.0,
            'equalized_odds_ratio': 0.0,
            'demographic_parity_difference': 0.0,
            'demographic_parity_ratio': 0.0
        }
    
    return {
        'group_aucs': group_aucs,
        'overall_auc': overall_auc,
        'min_auc': min_auc,
        'max_auc': max_auc,
        'auc_gap': auc_gap,
        'fairlearn_metrics': fairlearn_metrics
    }

def print_evaluation_results(results):
    print("Fairness Evaluation Results:")
    print(f"Overall AUC: {results['overall_auc']:.6f}")
    print(f"Min Group AUC: {results['min_auc']:.6f}")
    print(f"Max Group AUC: {results['max_auc']:.6f}")
    print(f"AUC Gap: {results['auc_gap']:.6f}")
    print("Group-wise AUCs:")
    for group_id, auc in results['group_aucs'].items():
        print(f"  Group {group_id}: {auc:.6f}")
    
    # Print fairlearn metrics
    if 'fairlearn_metrics' in results:
        print("\nFairlearn Metrics:")
        fl_metrics = results['fairlearn_metrics']
        print(f"Equalized Odds Difference: {fl_metrics['equalized_odds_difference']}")
        print(f"Equalized Odds Ratio: {fl_metrics['equalized_odds_ratio']}")
        print(f"Demographic Parity Difference: {fl_metrics['demographic_parity_difference']}")
        print(f"Demographic Parity Ratio: {fl_metrics['demographic_parity_ratio']}")