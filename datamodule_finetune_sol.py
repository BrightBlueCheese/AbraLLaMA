import lightning as L
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorWithPadding


class CustomLlamaDatasetAbraham(Dataset):
    def __init__(self, df, tokenizer, max_seq_length):
        self.keys = df.iloc[:, 0] # 1D array
        self.labels = df.iloc[:, 1:] # 2D array
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self):
        return self.keys.shape[0]

    def fn_token_encode(self, smiles):
        return self.tokenizer(
            smiles,
            truncation=True,
            padding="max_length",
            max_length=self.max_seq_length,
        )

    def __getitem__(self, idx):
        local_encoded = self.fn_token_encode(self.keys.iloc[idx])
        
        return {
            "input_ids": torch.tensor(local_encoded["input_ids"]),
            "attention_mask": torch.tensor(local_encoded["attention_mask"]),
            "labels": torch.tensor(self.labels.iloc[idx]),
        }

def load_abraham(
    solute_or_solvent:str,
    train_only:bool=False,
):
    assert solute_or_solvent in ['solvent', 'solute'], "'solute_or_solvent' can only be those following : ['solvent', 'solute']"
    if solute_or_solvent == 'solvent':
        dir_df = './dataset_ft/AbrahamSolvent_cleared.csv'
    elif solute_or_solvent == 'solute':
        dir_df = './dataset_ft/AbrahamSolute_cleared.csv'

    df = pd.read_csv(dir_df)

    if train_only == False:
        df_train, df_temp = train_test_split(df, test_size=0.3)
        df_valid, df_test = train_test_split(df_temp, test_size=0.5)
        
        return (df_train, df_valid, df_test)
    elif train_only == True:
        return (df, df, df) # valid and test sets are just nominal

class CustomFinetuneDataModule(L.LightningDataModule):
    def __init__(
        self,
        solute_or_solvent,
        tokenizer,
        max_seq_length,
        batch_size_train,
        batch_size_valid,
        num_device,
        train_only:bool=False,
    ):
        super().__init__()

        self.solute_or_solvent = solute_or_solvent
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.batch_size_train = batch_size_train
        self.batch_size_valid = batch_size_valid
        self.data_collator = DataCollatorWithPadding(self.tokenizer)
        self.num_device = num_device
        self.train_only = train_only
    

    def prepare_data(self):
        self.list_df = load_abraham(self.solute_or_solvent, self.train_only)
        # self.smiles_str

    def setup(self, stage=None):
        self.train_df, self.valid_df, self.test_df = self.list_df

    def train_dataloader(self):
        return DataLoader(
            dataset=CustomLlamaDatasetAbraham(
                self.train_df, self.tokenizer, self.max_seq_length,
            ),
            batch_size=self.batch_size_train,
            num_workers=self.num_device,
            collate_fn=self.data_collator,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=CustomLlamaDatasetAbraham(
                self.valid_df, self.tokenizer, self.max_seq_length,
            ),
            batch_size=self.batch_size_valid,
            num_workers=self.num_device,
            collate_fn=self.data_collator,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=CustomLlamaDatasetAbraham(
                self.test_df, self.tokenizer, self.max_seq_length,
            ),
            batch_size=self.batch_size_valid,
            num_workers=self.num_device,
            collate_fn=self.data_collator,
            shuffle=False,
        )
        
    # It uses test_df
    def predict_dataloader(self): 
        return DataLoader(
            dataset=CustomLlamaDatasetAbraham(
                self.test_df, self.tokenizer, self.max_seq_length,
            ),
            batch_size=self.batch_size_valid,
            num_workers=self.num_device,
            collate_fn=self.data_collator,
            shuffle=False,
        )

class CustomFinetuneDataModuleKfold(L.LightningDataModule):
    def __init__(
        self,
        solute_or_solvent,
        tokenizer,
        max_seq_length,
        batch_size_train,
        batch_size_valid,
        num_device,
    ):
        super().__init__()

        self.solute_or_solvent = solute_or_solvent
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.batch_size_train = batch_size_train
        self.batch_size_valid = batch_size_valid
        self.data_collator = DataCollatorWithPadding(self.tokenizer)
        self.num_device = num_device
    

    def prepare_data(self):
        self.list_df = load_abraham(self.solute_or_solvent)

    def setup(self, stage=None):
        self.train_df, self.valid_df, self.test_df = self.list_df

    def train_dataloader(self):
        return DataLoader(
            dataset=CustomLlamaDatasetAbraham(
                self.train_df, self.tokenizer, self.max_seq_length,
            ),
            batch_size=self.batch_size_train,
            num_workers=self.num_device,
            collate_fn=self.data_collator,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=CustomLlamaDatasetAbraham(
                self.valid_df, self.tokenizer, self.max_seq_length,
            ),
            batch_size=self.batch_size_valid,
            num_workers=self.num_device,
            collate_fn=self.data_collator,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=CustomLlamaDatasetAbraham(
                self.test_df, self.tokenizer, self.max_seq_length,
            ),
            batch_size=self.batch_size_valid,
            num_workers=self.num_device,
            collate_fn=self.data_collator,
            shuffle=False,
        )
        
    # It uses test_df
    def predict_dataloader(self): 
        return DataLoader(
            dataset=CustomLlamaDatasetAbraham(
                self.test_df, self.tokenizer, self.max_seq_length,
            ),
            batch_size=self.batch_size_valid,
            num_workers=self.num_device,
            collate_fn=self.data_collator,
            shuffle=False,
        )








        