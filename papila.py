import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
from sklearn.model_selection import GroupShuffleSplit
from config import Config

class PAPILADataset(Dataset):
    def __init__(self, dataframe, data_path, sensitive_attr='Sex_binary', transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.data_path = data_path
        self.sensitive_attr = sensitive_attr
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Filename format: RET<PatientID><Eye>.jpg
        # PatientID should be 3 digits, e.g. 002
        # Assuming 'ID' column exists and is integer
        pat_id = int(row['ID'])
        eye = row['Eye'] # 'OD' or 'OS'
        
        img_filename = f"RET{pat_id:03d}{eye}.jpg"
        
        # Images are stored in 'images' folder at data_path
        img_path = os.path.join(self.data_path, 'images', img_filename)
        
        # Fallback to root if not in images
        if not os.path.exists(img_path):
             img_path = os.path.join(self.data_path, img_filename)

        try:
            image = Image.open(img_path).convert('RGB')
        except (FileNotFoundError, OSError) as e:
            print(f"Error loading image: {img_path}")
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

def get_papila_transforms(is_training=True):
    """
    Get transforms for PAPILA dataset.
    """
    # Standard ImageNet normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if is_training:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            normalize
        ])
    else:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])

def prepare_papila_data(data_path):
    """
    Load and preprocess PAPILA metadata.
    """
    od_path = os.path.join(data_path, 'patient_data_od.xlsx')
    os_path = os.path.join(data_path, 'patient_data_os.xlsx')
    
    if not os.path.exists(od_path) or not os.path.exists(os_path):
        # Fallback to local
        od_path = 'patient_data_od.xlsx'
        os_path = 'patient_data_os.xlsx'
        
    print(f"Loading metadata from {od_path} and {os_path}...")
    
    # Load Excel files
    # Requires openpyxl
    try:
        df_od = pd.read_excel(od_path)
        df_os = pd.read_excel(os_path)
    except ImportError:
        raise ImportError("Reading Excel files requires 'openpyxl'. Please install it via pip.")
        
    # Add Eye column
    df_od['Eye'] = 'OD'
    df_os['Eye'] = 'OS'
    
    # Concatenate
    df = pd.concat([df_od, df_os], ignore_index=True)
    print(f"Total samples loaded: {len(df)}")
    
    # Check columns
    print("Columns found:", df.columns.tolist())
    
    # 1. Filter Diagnosis (0: Normal, 1: Glaucoma, 2: Suspect)
    # Exclude Suspect (2)
    if 'Diagnosis' in df.columns:
        df = df[df['Diagnosis'] != 2]
        df['binaryLabel'] = df['Diagnosis'].astype(int)
    else:
        raise KeyError("Column 'Diagnosis' not found in metadata.")
        
    print(f"Samples after excluding Suspects: {len(df)}")
    
    # 2. Sensitive Attributes
    # Sex_binary
    # Check for Gender or Sex
    if 'Gender' in df.columns:
        # Assuming Male/Female strings. Need to verify mapping.
        # Common: Male=1, Female=0 or vice versa.
        # Let's check unique values
        uniques = df['Gender'].unique()
        print(f"Gender values: {uniques}")
        # Map Male to 1, Female to 0 (Consistent with HAM10000)
        df['Sex_binary'] = df['Gender'].apply(lambda x: 1 if str(x).lower().startswith('m') else 0)
    elif 'Sex' in df.columns:
        uniques = df['Sex'].unique()
        print(f"Sex values: {uniques}")
        df['Sex_binary'] = df['Sex'].apply(lambda x: 1 if str(x).lower().startswith('m') else 0)
    else:
        print("Warning: Gender/Sex column not found. Creating dummy Sex_binary.")
        df['Sex_binary'] = 0
        
    # Age_binary
    if 'Age' in df.columns:
        df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
        df['Age_binary'] = (df['Age'] > 60).astype(int)
    else:
        print("Warning: Age column not found. Creating dummy Age_binary.")
        df['Age_binary'] = 0
        
    # 3. Patient-level Splitting
    # 70% Train, 10% Val, 20% Test
    # Ensure 'ID' column exists
    if 'ID' not in df.columns:
        # Try to find ID column
        possible_ids = [c for c in df.columns if 'id' in c.lower()]
        if possible_ids:
            print(f"Using {possible_ids[0]} as ID column")
            df['ID'] = df[possible_ids[0]]
        else:
            raise KeyError("Could not identify Patient ID column for splitting.")
            
    splitter = GroupShuffleSplit(n_splits=1, train_size=0.7, random_state=42)
    train_idx, temp_idx = next(splitter.split(df, groups=df['ID']))
    
    train_df = df.iloc[train_idx]
    temp_df = df.iloc[temp_idx]
    
    # Split Temp into Val (1/3) and Test (2/3) -> 10% and 20% of total
    # 10% is 1/3 of 30%
    splitter_val = GroupShuffleSplit(n_splits=1, test_size=2/3, random_state=42)
    val_idx, test_idx = next(splitter_val.split(temp_df, groups=temp_df['ID']))
    
    val_df = temp_df.iloc[val_idx]
    test_df = temp_df.iloc[test_idx]
    
    print(f"Split sizes (Patient-level):")
    print(f"Train: {len(train_df)} ({len(train_df['ID'].unique())} patients)")
    print(f"Val:   {len(val_df)} ({len(val_df['ID'].unique())} patients)")
    print(f"Test:  {len(test_df)} ({len(test_df['ID'].unique())} patients)")
    
    # Create Datasets
    sensitive_attr = Config.dataset_config.get('sensitive_attr', 'Sex_binary')
    
    train_transform = get_papila_transforms(is_training=True)
    val_transform = get_papila_transforms(is_training=False)
    
    train_dataset = PAPILADataset(train_df, data_path, sensitive_attr=sensitive_attr, transform=train_transform)
    val_dataset = PAPILADataset(val_df, data_path, sensitive_attr=sensitive_attr, transform=val_transform)
    test_dataset = PAPILADataset(test_df, data_path, sensitive_attr=sensitive_attr, transform=val_transform)
    
    return train_dataset, val_dataset, test_dataset
