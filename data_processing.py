import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset



res_to_id = {
    "X": 0,
    "A": 1,
    "C": 2,
    "D": 3,
    "E": 4,
    "F": 5,
    "G": 6,
    "H": 7,
    "I": 8,
    "K": 9,
    "L": 10,
    "M": 11,
    "N": 12,
    "P": 13,
    "Q": 14,
    "R": 15,
    "S": 16,
    "T": 17,
    "V": 18,
    "W": 19,
    "Y": 20,
    "U": 2,
    "Z": 0
}



class ProteinDataset(Dataset):
    def __init__(self, dataframe, emb_path, target_aa):
        self.dataframe = dataframe
        self.target_aa = target_aa
        self.data_list = []
        df = dataframe
        for _, record in df.iterrows():
            seq = record['Sequence']
            uniprot = record['Entry']
            prot_len = record['Length']
            emb = torch.tensor(
                np.load(os.path.join(emb_path, "{}.npy".format(record['Entry']))), dtype=torch.float32)
            x = [res_to_id[res] for res in seq]
            if isinstance(record['Position'], int):
                positions = [record['Position'] - 1]  
            else:
                positions = [int(pos) - 1 for pos in record['Position'].split(', ')] 
            label = [1 if idx in positions else 0 for idx in range(len(seq))]
            label = torch.tensor(label, dtype=torch.long)
            mask = [1 if res == self.target_aa else 0 for res in seq]
            mask = torch.tensor(mask, dtype=torch.long)
  #          one_hot = F.one_hot(torch.tensor(x), num_classes=len(res_to_id))
            
            data = {
                'x': x,
                'emb': emb,
                'seq': seq,
                'uniprot': uniprot,
                'prot_len': prot_len,
                'label': label,
                'mask': mask,
             
            }

            self.data_list.append(data)


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]

        batch = {
            "embedding": item['emb'],
            "label": item['label'],
            "mask": item['mask'],
            "prot_len": item['prot_len'],
         
        }
        return batch
    
    def split(self, val_size=0.1):
        train_idx, val_idx = train_test_split(range(len(self)), test_size=val_size, random_state=42)
        train_subset = Subset(self, train_idx)
        val_subset = Subset(self, val_idx)
        return train_subset, val_subset

def collate_fn(batch):
    
    max_len = max(item['embedding'].shape[0] for item in batch)
    
    padded_embs = []
    padded_labels = []
    padded_masks = []

    
    for item in batch:
        current_len = item['embedding'].shape[0]
        
 
        padded_emb = torch.zeros(max_len, 1280) 
        padded_emb[:current_len, :] = item['embedding']
        padded_embs.append(padded_emb)
        

        padded_label = torch.zeros(max_len, dtype=torch.long)  # 使用适当的dtype
        padded_label[:current_len] = item['label']
        padded_labels.append(padded_label)
        

        padded_mask = torch.zeros(max_len, dtype=torch.bool)  # 假设mask是布尔类型
        padded_mask[:current_len] = item['mask']
        padded_masks.append(padded_mask)

    

    batch_embs = torch.stack(padded_embs)
    batch_labels = torch.stack(padded_labels)
    batch_masks = torch.stack(padded_masks)

    return {
        'embedding': batch_embs,
        'label': batch_labels,
        'mask': batch_masks 
    }






