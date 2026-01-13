import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
from sklearn.model_selection import train_test_split
from config import Config

class Fitzpatrick17kDataset(Dataset):
    def __init__(self, dataframe, data_path, sensitive_attr='Skin_type_binary', transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.data_path = data_path
        self.sensitive_attr = sensitive_attr
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Image filename is md5hash.jpg
        img_filename = f"{row['md5hash']}.jpg"
        
        # Images are stored in 'images' folder at data_path
        img_path = os.path.join(self.data_path, 'images', img_filename)

        try:
            image = Image.open(img_path).convert('RGB')
        except (FileNotFoundError, OSError) as e:
            print(f"Error loading image: {img_path}")
            # Return a black image to avoid crashing? Or raise?
            # Raising is better to know something is wrong.
            raise e

        if self.transform:
            image = self.transform(image)
        
        label = torch.tensor(int(row['binaryLabel']), dtype=torch.long)
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

def get_fitzpatrick_transforms(is_training=True):
    """
    Get transforms for Fitzpatrick17k dataset.
    """
    # Standard ImageNet normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    if is_training:
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),       
            transforms.ToTensor(),
            normalize
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),         
            transforms.ToTensor(),
            normalize
        ])

def prepare_fitzpatrick_data(data_path):
    """
    Load and preprocess Fitzpatrick17k metadata.
    """
    csv_path = os.path.join(data_path, 'fitzpatrick17k_metadata.csv')
    if not os.path.exists(csv_path):
        # Fallback to local path if running in a different env
        csv_path = 'fitzpatrick17k_metadata.csv'
        
    print(f"Loading metadata from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Ensure fitzpatrick_scale is numeric, coerce errors to NaN
    df['fitzpatrick_scale'] = pd.to_numeric(df['fitzpatrick_scale'], errors='coerce')
    
    # Drop NaNs in fitzpatrick_scale if any (treated as missing skin type)
    df = df.dropna(subset=['fitzpatrick_scale'])
    
    # 1. Filter out skin type -1
    print(f"Original samples: {len(df)}")
    df = df[df['fitzpatrick_scale'] != -1]
    print(f"Samples after removing skin type -1: {len(df)}")
    
    # 2. Create Binary Labels
    # 'benign' and 'non-neoplastic' -> 0 (Benign)
    # 'malignant' -> 1 (Malignant)
    def map_label(val):
        if val in ['benign', 'non-neoplastic']:
            return 0
        elif val == 'malignant':
            return 1
        else:
            return -1 # Should not happen based on description
            
    df['binaryLabel'] = df['three_partition_label'].apply(map_label)
    
    # Check for unmapped labels
    if (df['binaryLabel'] == -1).any():
        print("Warning: Found unknown labels in three_partition_label")
        print(df[df['binaryLabel'] == -1]['three_partition_label'].unique())
        df = df[df['binaryLabel'] != -1]
        
    # 3. Create Binary Sensitive Attribute (Skin Type)
    # Group 0: 1-3
    # Group 1: >3 (4-6)
    def map_skin_type(val):
        if val <= 3:
            return 0
        else:
            return 1
            
    df['Skin_type_binary'] = df['fitzpatrick_scale'].apply(map_skin_type)
    
    print("Label distribution:")
    print(df['binaryLabel'].value_counts())
    print("Skin Type Group distribution:")
    print(df['Skin_type_binary'].value_counts())
    
    # Split Data
    # 80% Train, 10% Val, 10% Test
    train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df['binaryLabel'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['binaryLabel'], random_state=42)
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Create Datasets
    train_transform = get_fitzpatrick_transforms(is_training=True)
    val_transform = get_fitzpatrick_transforms(is_training=False)
    
    train_dataset = Fitzpatrick17kDataset(train_df, data_path, transform=train_transform)
    val_dataset = Fitzpatrick17kDataset(val_df, data_path, transform=val_transform)
    test_dataset = Fitzpatrick17kDataset(test_df, data_path, transform=val_transform)
    
    return train_dataset, val_dataset, test_dataset
