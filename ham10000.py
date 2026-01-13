import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
from sklearn.model_selection import train_test_split
from config import Config

class HAM10000Dataset(Dataset):
    def __init__(self, dataframe, data_path, sensitive_attr='Sex_binary', transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.data_path = data_path
        self.sensitive_attr = sensitive_attr
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Construct image path
        img_path = os.path.join(self.data_path, row['Path'])
        
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

def get_transforms(is_training=True):
    """
    Get transforms for HAM10000 dataset.
    Corrects the previous implementation by using proper augmentation and normalization.
    """
    # Standard ImageNet normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # if is_training:
    #     return transforms.Compose([
    #         transforms.RandomResizedCrop(224, scale=(0.8, 1.0)), # Random crop with resizing
    #         transforms.RandomHorizontalFlip(),
    #         transforms.RandomVerticalFlip(), # Skin lesions don't have a natural orientation
    #         transforms.RandomRotation(15),
    #         transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
    #         transforms.ToTensor(),
    #         normalize
    #     ])
    # else:
    #     return transforms.Compose([
    #         transforms.Resize(256),
    #         transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #         normalize
    #     ])
    if is_training:
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.90, 1.0)), 
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(45),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),            
            transforms.ToTensor(),
            normalize
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),        
            transforms.ToTensor(),
            normalize
        ])

def preprocess_metadata(data_path):
    """
    Preprocess HAM10000 metadata.
    Fixes label mapping and creates binary columns.
    """
    csv_path = os.path.join(data_path, 'HAM10000_metadata.csv')
    if not os.path.exists(csv_path):
        # Fallback to checking if it's in the current directory if data_path is different
        if os.path.exists('HAM10000_metadata.csv'):
            csv_path = 'HAM10000_metadata.csv'
        else:
            raise FileNotFoundError(f"Metadata file not found at {csv_path} or current directory")
        
    print(f"Loading metadata from {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Create Path column
    # Assuming images are in an 'images' subdirectory
    df['Path'] = df['image_id'].apply(lambda x: os.path.join('images', f'{x}.jpg'))
    
    # Filter missing values
    df = df.dropna(subset=['age', 'sex'])
    # Filter unknown sex
    df = df[df['sex'] != 'unknown']
    
    # Binary Sex
    df['Sex_binary'] = (df['sex'] == 'male').astype(int)
    
    # Binary Label (Benign vs Malignant)
    # Malignant: mel (Melanoma), akiec (Actinic keratoses)
    # Benign: nv (Melanocytic nevi), bkl (Benign keratosis-like lesions), 
    #         vasc (Vascular lesions), df (Dermatofibroma), bcc (Basal cell carcinoma)
    # Treat common malignancies as positive
    malignant_classes = ['mel', 'akiec']
    df['binaryLabel'] = df['dx'].apply(lambda x: 1 if x in malignant_classes else 0)
    
    # Age binary: 0 for age <= 60, 1 for age > 60
    # Cast to numeric if possible, otherwise mark as NaN and drop above
    try:
        ages = pd.to_numeric(df['age'], errors='coerce')
        df['Age_binary'] = (ages > 60).astype(int)
    except Exception:
        df['Age_binary'] = 0
    
    print("Data Preprocessing Summary:")
    print(f"Total samples: {len(df)}")
    print(f"Malignant samples: {df['binaryLabel'].sum()}")
    print(f"Benign samples: {len(df) - df['binaryLabel'].sum()}")
    
    sensitive_attr = Config.dataset_config.get('sensitive_attr', 'Sex_binary')
    if sensitive_attr in df.columns:
        print(f"{sensitive_attr} distribution: \n{df[sensitive_attr].value_counts()}")
    
    return df

def get_splits(df, test_size=0.2, val_size=0.5, seed=42, sensitive_attr='Sex_binary'):
    """
    Create stratified splits.
    """
    # Stratified split based on Sex and Label to ensure balanced representation
    df = df.copy()
    if sensitive_attr not in df.columns:
        raise KeyError(f"Sensitive attribute '{sensitive_attr}' not found in dataframe columns")
    df['stratify_col'] = df[sensitive_attr].astype(str) + '_' + df['binaryLabel'].astype(str)
    
    train_df, temp_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df['stratify_col']
    )
    
    val_df, test_df = train_test_split(
        temp_df, test_size=val_size, random_state=seed, stratify=temp_df['stratify_col']
    )
    
    # Clean up
    for d in [train_df, val_df, test_df]:
        d.drop(columns=['stratify_col'], inplace=True)
        
    print(f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    return train_df, val_df, test_df

def prepare_ham_data(data_path, seed=42):
    """
    Main entry point to get datasets.
    """
    df = preprocess_metadata(data_path)
    sensitive_attr = Config.dataset_config.get('sensitive_attr', 'Sex_binary')
    
    # Fallback check
    if sensitive_attr not in df.columns:
        print(f"⚠️ Warning: Sensitive attribute '{sensitive_attr}' not found in HAM10000 dataset. Falling back to 'Sex_binary'.")
        sensitive_attr = 'Sex_binary'
        
    train_df, val_df, test_df = get_splits(df, seed=seed, sensitive_attr=sensitive_attr)

    train_dataset = HAM10000Dataset(train_df, data_path, sensitive_attr=sensitive_attr, transform=get_transforms(is_training=True))
    val_dataset = HAM10000Dataset(val_df, data_path, sensitive_attr=sensitive_attr, transform=get_transforms(is_training=False))
    test_dataset = HAM10000Dataset(test_df, data_path, sensitive_attr=sensitive_attr, transform=get_transforms(is_training=False))

    return train_dataset, val_dataset, test_dataset
