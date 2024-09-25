import time
start_time = time.time()

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
import torch
from torch.utils.data import DataLoader
from Data.data_processing import ProteinDataset, collate_fn
from types import SimpleNamespace
import pandas as pd
import numpy as np
import umap

class TransformerModel(nn.Module):
    def __init__(self, emb_size, num_heads, num_layers, hidden_size, dropout_rate, pos_weight=None):
        super(TransformerModel, self).__init__()
        self.cnn = nn.Conv1d(in_channels=emb_size, out_channels=256, kernel_size=31, padding=15) 
        self.transformer = nn.TransformerEncoderLayer(d_model=256, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.transformer, num_layers=num_layers)
        self.fc1 = nn.Linear(256, hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.pos_weight = pos_weight

    def forward(self, batch):
        emb = batch['embedding']
        emb = emb.transpose(1, 2)  # assuming emb is shape (batch, emb_dim, seq_length)
        emb = self.cnn(emb)
        emb = emb.transpose(1, 2)  # switch back to (batch, seq_length, emb_dim)
        x = self.encoder(emb)
        return x

    def get_site_features(self, batch):
        # Assuming batch['mask'] is a boolean tensor where 1 indicates serine sites
        encoder_output = self(batch)
        mask = batch['mask'].bool()  # ensure mask is boolean
        indices = mask.nonzero(as_tuple=True)  # Get indices where mask is True

        # Fetch encoder outputs only for serine positions
        features = encoder_output[indices]

        # Optional: Collect labels for these serine positions if needed
        labels = batch['label'][mask]

        # Return a dictionary with serine features and their labels
        return {
            'features': features,
            'labels': labels
        },encoder_output


# 初始化模型
model = TransformerModel(emb_size=1280, num_heads=8, num_layers=1, hidden_size=128, dropout_rate=0.5)
model.to('cpu')  

model_path = '/proj/results_Phosphoserine/emb_type_esm2_epochs_200_batch_32_opt_adam_lr_5e-05_patience_40_dropout_0.5_posWeight_3/0_acc0.9384_precision_0.6456_recall_0.6145_f1_0.6297_mcc_0.5963_roc0.9096_prc0.6755.pt'
model.load_state_dict(torch.load(model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu'), strict=True)

EMB_PATH = f'./Data/esm2_all'
data_path = '/proj/Data/Phosphoserine_clustered_splited.csv'
data = pd.read_csv(data_path)
train_data = data[data['Set'] == 'test']
train_dataset = ProteinDataset(train_data, EMB_PATH, target_aa='S')
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)


model.eval()

all_features = []
all_labels = []

for batch in train_dataloader:
    site_features, full_encoder_output = model.get_site_features(batch)
    
    for i in range(site_features['features'].shape[0]):
        feature = site_features['features'][i].detach().numpy()  
        label = site_features['labels'][i].item()  
        all_features.append(feature)
        all_labels.append(label)


all_features = np.array(all_features)
all_labels = np.array(all_labels)


import matplotlib.pyplot as plt

umap_reducer = umap.UMAP(n_neighbors=15, min_dist=1, n_components=2, random_state=42)
transformed_features = umap_reducer.fit_transform(all_features)


plt.figure(figsize=(10, 8))
for label in np.unique(all_labels):
    indices = all_labels == label
    plt.scatter(transformed_features[indices, 0], transformed_features[indices, 1], label=f"Label {label}",s=1)

plt.legend()
plt.title('UMAP Visualization of Protein Features')
plt.xlabel('UMAP Component 1')
plt.ylabel('UMAP Component 2')

plt.savefig('UMAP_visualization.png', format='png', dpi=300)
plt.show()

end_time = time.time()
print(f"Total execution time: {end_time - start_time:.2f} seconds")
