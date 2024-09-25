import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset


EMB_PATH = './Data/esm2'


res_to_id = {
    "*": 0,
    "K": 1,
    "A": 2,
    "R": 3,
    "N": 4,
    "D": 5,
    "C": 6,
    "Q": 7,
    "E": 8,
    "G": 9,
    "H": 10,
    "I": 11,
    "L": 12,
    "M": 13,
    "F": 14,
    "P": 15,
    "S": 16,
    "T": 17,
    "W": 18,
    "Y": 19,
    "V": 20,
    "U": 6
}


class Processor:
    def __init__(self, info_df):
        self.info = info_df


    def run(self):
        train_ls = []
        test_ls = []
        for _, record in self.info.iterrows():
            seq = record['Sequence']
            uniprot = record['Entry']
            prot_len = record['Length']
            mask = record['Mask']
            emb = torch.tensor(
                np.load(os.path.join(EMB_PATH, "{}.npy".format(record['Entry']))), dtype=torch.float32)
            x = [res_to_id[res] for res in seq]
            label = torch.from_numpy(np.fromstring(record['Label'][1:-1], sep=',')).long()
            mask = torch.from_numpy(np.fromstring(record['Mask'][1:-1], sep=',')).long()

            data ={
                'x' : x,
                'emb' : emb,
                'seq' : seq,
                'uniprot' : uniprot,
                'prot_len': prot_len,
                'label' : label,
                'mask' : mask
            }
            group = record['Set']
            if group == 'train':
                train_ls.append(data)
            elif group == 'test':
                test_ls.append(data)
            else:
                raise Exception('Unknown data group')

        return train_ls, test_ls




if __name__ == '__main__':
    df = pd.read_csv('./Data/dataset.csv')
    save_path = './Data/full_length.pt'
    processor = Processor(df)
    data_ls = processor.run()
    torch.save(data_ls, save_path)
