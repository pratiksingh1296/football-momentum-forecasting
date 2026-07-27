# Standard library
from pathlib import Path
import sys
import json

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Third-party
import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer
import joblib

# Local modules
from debug import debug_print
from config import *

class MomentumDataset(Dataset):

    def __init__(self, csv_file, tokenizer, experiment='baseline', max_length=128):

        DATA_PATH = (BASE_DIR / "data" / "processed" / experiment)

        self.data = pd.read_csv(DATA_PATH / csv_file)
        self.labels = self.data['label'].values
        self.encodings = tokenizer(
            self.data['text'].tolist(),
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        debug_print(f"Loaded {len(self.data)} samples from {csv_file}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return {
            'input_ids': self.encodings['input_ids'][index],
            'attention_mask': self.encodings['attention_mask'][index],
            'label': torch.tensor(self.labels[index], dtype=torch.long)
        }


class HybridMomentumDataset(Dataset):

    def __init__(self, parquet_file, tokenizer, experiment='exp_d_hybrid', max_length = 128):

        DATA_PATH = BASE_DIR / "data" / "processed" / experiment
        MODELS_PATH = BASE_DIR / "models" / experiment

        self.data = pd.read_parquet(DATA_PATH / parquet_file)
        self.labels = self.data['label'].values

        # Loading fitted scalar 
        self.scaler = joblib.load(MODELS_PATH / 'numeric_scaler.joblib')
        with open(MODELS_PATH / 'numeric_cols.json') as f:
            self.numeric_cols = json.load(f)

        # Missing Check
        missing = set(self.numeric_cols) - set(self.data.columns)
        if missing:
            raise ValueError(
                f"Missing numeric columns: {missing}"
            )

        # Applying scaling using fitted scalar
        numeric_array = self.scaler.transform(self.data[self.numeric_cols])
        assert numeric_array.shape[1] == len(self.numeric_cols)
        self.numeric_features = torch.tensor(numeric_array, dtype=torch.float32)

        

        self.encodings = tokenizer(
            self.data['text'].tolist(),
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        debug_print(f"Loaded {len(self.data)} samples from {parquet_file}, {len(self.numeric_cols)} numeric features")

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        return {
            'input_ids': self.encodings['input_ids'][index],
            'attention_mask': self.encodings['attention_mask'][index],
            'numeric_features': self.numeric_features[index],
            'label': torch.tensor(self.labels[index], dtype=torch.long)
        }
    

# Test Block
if __name__ == "__main__":

    '''
    # Test MomentumDataset
    print('MomentumDataset: \n')
    print('*' * 50)
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    dataset = MomentumDataset('train.csv', tokenizer)
    sample = dataset[0]
    debug_print(sample['input_ids'].shape)
    debug_print(sample['attention_mask'].shape)
    debug_print(sample['label'])
    '''
    
    # Test HybridMomentumDataset
    print('HybridMomentumDataset: \n')
    print('*' * 50)
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    dataset = HybridMomentumDataset('train.parquet', tokenizer=tokenizer)

    sample = dataset[0]
    print("input_ids:", sample['input_ids'].shape)
    print("attention_mask:", sample['attention_mask'].shape)
    print("numeric_features:", sample['numeric_features'].shape)
    print("label:", sample['label'])
    print("numeric_features values:", sample['numeric_features'])