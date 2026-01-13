import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
from sklearn.model_selection import train_test_split
from config import Config
import h5py

class OL3IDataset(Dataset):
    def __init__(self, dataframe, data_path, sensitive_attr='Sex_binary', transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.data_path = data_path
        self.sensitive_attr = sensitive_attr
        self.transform = transform
        self.h5_path = os.path.join(data_path, 'l3_slices.h5')
        self.h5_file = None

    def __len__(self):
        return len(self.df)

    def _open_h5(self):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')

    def __getitem__(self, idx):
        if self.h5_file is None:
            self._open_h5()
            
        row = self.df.iloc[idx]
        anon_id = row['anon_id']
        
        try:
            # Read image from HDF5
            # Dataset is named with the hash
            if anon_id in self.h5_file:
                image_data = self.h5_file[anon_id][:]
                # Assuming image is 512x512, might be grayscale or RGB?
                # CT slices are usually grayscale (1 channel) or stored as numpy arrays.
                # If it's a numpy array, we need to convert to PIL Image for transforms.
                
                # Check dimensions
                if len(image_data.shape) == 2:
                    # Grayscale 512x512
                    image = Image.fromarray(image_data).convert('RGB') # Convert to RGB for ViT compatibility
                elif len(image_data.shape) == 3:
                    image = Image.fromarray(image_data)
                else:
                    raise ValueError(f"Unexpected image shape: {image_data.shape}")
            else:
                # Fallback or error
                print(f"Warning: ID {anon_id} not found in HDF5")
                # Return a black image or raise error?
                # For now, raise error to be safe
                raise FileNotFoundError(f"ID {anon_id} not found in {self.h5_path}")

        except Exception as e:
            print(f"Error loading image for ID {anon_id}: {e}")
            raise e

        if self.transform:
            image = self.transform(image)
        
        label = torch.tensor(int(row['label_1y']), dtype=torch.long)
        group = torch.tensor(int(row[self.sensitive_attr]), dtype=torch.long)
        
        return {'image': image, 'label': label, 'group': group}

    def get_groups(self):
        """
        Extract groups for FairDI training.
        Returns a dictionary {group_id: [indices]}
        """
        groups = {}
        # Use the dataframe for fast group extraction
        group_column = self.df[self.sensitive_attr].values
        for idx, gid in enumerate(group_column):
            gid = int(gid)
            groups.setdefault(gid, []).append(idx)
            
        print(f'Found {len(groups)} demographic groups')
        print(f'Total samples: {len(self)}')
        for gid, indices in groups.items():
            print(f'Group {gid}: {len(indices)} samples')
        return groups
    
    def __del__(self):
        if self.h5_file is not None:
            self.h5_file.close()

def get_ol3i_transforms(is_training=True):
    """
    Get transforms for OL3I dataset.
    """
    # Standard ImageNet normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    if is_training:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            normalize
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),            
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            normalize
        ])
    
def preprocess_ol3i_metadata(data_path):
    """
    Preprocess OL3I metadata.
    """
    csv_path = os.path.join(data_path, 'ol3i_metadata.csv')
    if not os.path.exists(csv_path):
        if os.path.exists('ol3i_metadata.csv'):
            csv_path = 'ol3i_metadata.csv'
        else:
            raise FileNotFoundError(f"Metadata file not found at {csv_path}")
        
    print(f"Loading metadata from {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Filter missing labels or sex
    # We are predicting label_1y
    df = df.dropna(subset=['label_1y', 'sex'])
    
    # Filter unknown sex if any (though CSV showed male/female)
    df = df[df['sex'].isin(['male', 'female'])]
    
    # Binary Sex
    df['Sex_binary'] = (df['sex'] == 'male').astype(int)
    
    # Age binary: 0 for age <= 60, 1 for age > 60
    # Cast to numeric if possible, otherwise mark as NaN
    try:
        ages = pd.to_numeric(df['age'], errors='coerce')
        df['Age_binary'] = (ages > 60).astype(int)
    except Exception:
        df['Age_binary'] = 0

    # Ensure label is integer
    df['label_1y'] = df['label_1y'].astype(int)
    
    print("Data Preprocessing Summary (OL3I):")
    print(f"Total samples: {len(df)}")
    print(f"Positive samples (1y): {df['label_1y'].sum()}")
    print(f"Negative samples (1y): {len(df) - df['label_1y'].sum()}")
    
    sensitive_attr = Config.dataset_config.get('sensitive_attr', 'Sex_binary')
    if sensitive_attr in df.columns:
        print(f"{sensitive_attr} distribution: \n{df[sensitive_attr].value_counts()}")
    else:
        print(f"Sex distribution: \n{df['sex'].value_counts()}")
    
    return df

def get_ol3i_splits(df, train_size=0.8, val_size=0.1, test_size=0.1, seed=42, sensitive_attr='Sex_binary'):
    """
    Create stratified splits: 80/10/10
    """
    # Normalize sizes
    total = train_size + val_size + test_size
    train_ratio = train_size / total
    val_ratio = val_size / total
    test_ratio = test_size / total
    
    df = df.copy()
    # Stratify by sensitive attribute and label
    df['stratify_col'] = df[sensitive_attr].astype(str) + '_' + df['label_1y'].astype(str)
    
    # First split: Train vs (Val + Test)
    # Train = 80%, Temp = 20%
    train_df, temp_df = train_test_split(
        df, test_size=(1 - train_ratio), random_state=seed, stratify=df['stratify_col']
    )
    
    # Second split: Val vs Test
    # Val is 10% of total, Test is 10% of total.
    # Temp is 20% of total.
    val_fraction = val_ratio / (val_ratio + test_ratio)
    
    val_df, test_df = train_test_split(
        temp_df, test_size=(1 - val_fraction), random_state=seed, stratify=temp_df['stratify_col']
    )
    
    # Clean up
    for d in [train_df, val_df, test_df]:
        d.drop(columns=['stratify_col'], inplace=True)
        
    print(f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    return train_df, val_df, test_df

def prepare_ol3i_data(data_path, seed=42):
    """
    Main entry point to get OL3I datasets.
    """
    df = preprocess_ol3i_metadata(data_path)
    
    # Use sensitive attribute from Config
    sensitive_attr = Config.dataset_config.get('sensitive_attr', 'Sex_binary')
    
    # Fallback check
    if sensitive_attr not in df.columns:
        print(f"⚠️ Warning: Sensitive attribute '{sensitive_attr}' not found in OL3I dataset. Falling back to 'Sex_binary'.")
        sensitive_attr = 'Sex_binary'
        
    print(f"Using sensitive attribute: {sensitive_attr}")
    
    train_df, val_df, test_df = get_ol3i_splits(df, seed=seed, sensitive_attr=sensitive_attr)

    train_dataset = OL3IDataset(train_df, data_path, sensitive_attr=sensitive_attr, transform=get_ol3i_transforms(is_training=True))
    val_dataset = OL3IDataset(val_df, data_path, sensitive_attr=sensitive_attr, transform=get_ol3i_transforms(is_training=False))
    test_dataset = OL3IDataset(test_df, data_path, sensitive_attr=sensitive_attr, transform=get_ol3i_transforms(is_training=False))

    return train_dataset, val_dataset, test_dataset
