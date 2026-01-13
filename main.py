import pandas as pd
import numpy as np
import torch
import os
import pickle
from run_full_pipeline import train_fairdi_complete
from config import Config
from ham10000 import prepare_ham_data
try:
    from ol3i import prepare_ol3i_data
except ImportError:
    print("Warning: Could not import ol3i module. OL3I dataset will not be available.")
    prepare_ol3i_data = None
try:
    from fitzpatrick17k import prepare_fitzpatrick_data
except ImportError:
    print("Warning: Could not import fitzpatrick17k module. FITZPATRICK17K dataset will not be available.")
    prepare_fitzpatrick_data = None
try:
    from papila import prepare_papila_data
except ImportError:
    print("Warning: Could not import papila module. PAPILA dataset will not be available.")
    prepare_papila_data = None

# Disable warnings and multiprocessing
import warnings
warnings.filterwarnings('ignore')
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def print_dataset_statistics(train_dataset, val_dataset, test_dataset):
    print("\n" + "="*65)
    print("DATASET STATISTICS (Group x Label Distribution)")
    print("="*65)
    
    splits = [('Train', train_dataset), ('Val', val_dataset), ('Test', test_dataset)]
    
    # Header
    print(f"{'Split':<8} | {'Group':<6} | {'Label 0':<8} | {'Label 1':<8} | {'Total':<8} | {'% Pos':<6}")
    print("-" * 65)
    
    for split_name, dataset in splits:
        if dataset is None:
            continue
            
        # Try to get dataframe
        if hasattr(dataset, 'df'):
            df = dataset.df
            
            # Identify label column
            label_col = None
            if 'binaryLabel' in df.columns:
                label_col = 'binaryLabel'
            elif 'label_1y' in df.columns:
                label_col = 'label_1y'
            
            # Identify group column
            group_col = getattr(dataset, 'sensitive_attr', None)
            if group_col is None or group_col not in df.columns:
                # Fallback or try to guess
                if 'Sex_binary' in df.columns:
                    group_col = 'Sex_binary'
            
            if label_col and group_col:
                groups = sorted(df[group_col].unique())
                
                split_total_0 = 0
                split_total_1 = 0
                split_total = 0
                
                for i, group in enumerate(groups):
                    group_df = df[df[group_col] == group]
                    n_0 = len(group_df[group_df[label_col] == 0])
                    n_1 = len(group_df[group_df[label_col] == 1])
                    total = n_0 + n_1
                    pos_rate = (n_1 / total * 100) if total > 0 else 0
                    
                    split_total_0 += n_0
                    split_total_1 += n_1
                    split_total += total
                    
                    prefix = split_name if i == 0 else ""
                    print(f"{prefix:<8} | {group:<6} | {n_0:<8} | {n_1:<8} | {total:<8} | {pos_rate:.1f}%")
                
                total_pos_rate = (split_total_1 / split_total * 100) if split_total > 0 else 0
                print(f"{'':<8} | {'Total':<6} | {split_total_0:<8} | {split_total_1:<8} | {split_total:<8} | {total_pos_rate:.1f}%")
                print("-" * 65)
            else:
                print(f"{split_name:<8} | Could not identify label/group columns")
        else:
             print(f"{split_name:<8} | Dataset does not have .df attribute")
    print("="*65 + "\n")

def get_datasets(dataset_name, data_path):
    """
    Factory function to get datasets based on name.
    """
    if dataset_name == 'HAM10000':
        print("Loading HAM10000 dataset...")
        # HAM10000 module handles all preprocessing and splitting internally
        return prepare_ham_data(data_path)
    elif dataset_name == 'OL3I':
        print("Loading OL3I dataset...")
        if prepare_ol3i_data is None:
             raise ImportError("OL3I module not found.")
        return prepare_ol3i_data(data_path)
    elif dataset_name == 'FITZPATRICK17K':
        print("Loading FITZPATRICK17K dataset...")
        if prepare_fitzpatrick_data is None:
             raise ImportError("FITZPATRICK17K module not found.")
        return prepare_fitzpatrick_data(data_path)
    elif dataset_name == 'PAPILA':
        print("Loading PAPILA dataset...")
        if prepare_papila_data is None:
             raise ImportError("PAPILA module not found.")
        return prepare_papila_data(data_path)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Please implement a loader in get_datasets.")

def save_models_and_results(teachers, student, results, save_dir='saved_models'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}")
    
    print("Saving teacher models...")
    for gid, teacher in teachers.items():
        torch.save(teacher.state_dict(), os.path.join(save_dir, f'teacher_group_{gid}.pth'))
        print(f"Saved teacher for group {gid}")
    
    print("Saving student model...")
    torch.save(student.state_dict(), os.path.join(save_dir, 'student.pth'))
    
    print("Saving evaluation results...")
    with open(os.path.join(save_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    print(f"All models and results saved to: {save_dir}")
    return save_dir

def main():
    set_seed(42)
    data_path = Config.data_path
    dataset_name = Config.dataset_config.get('name', 'HAM10000')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using GPU" if device == 'cuda' else "Using CPU")
    print(f"Data path: {data_path}")
    print(f"Dataset: {dataset_name}")
    
    # Get datasets
    try:
        train_dataset, val_dataset, test_dataset = get_datasets(dataset_name, data_path)
        print_dataset_statistics(train_dataset, val_dataset, test_dataset)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print("Starting FairDI pipeline...")
    teachers, student, results = train_fairdi_complete(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        num_classes=Config.dataset_config.get('num_classes', 2),
        device=device,
        backbone='vit_base_patch16_224',
        val_dataset=val_dataset
    )
    
    print("Training completed!")
    save_dir = save_models_and_results(teachers, student, results)
    print(f"Pipeline completed successfully! Models saved in: {save_dir}")
    
    return teachers, student, results

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    torch.set_num_threads(1)
    main()
