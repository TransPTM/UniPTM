import time
start_time = time.time()

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
from Data.data_processing import ProteinDataset, collate_fn
from types import SimpleNamespace
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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



model = TransformerModel(emb_size=1280, num_heads=8, num_layers=1, hidden_size=128, dropout_rate=0.5)
model.to('cpu')  

model_path = '/proj/results_Phosphoserine/emb_type_esm2_epochs_200_batch_32_opt_adam_lr_5e-05_patience_40_dropout_0.5_posWeight_3/2_acc0.9371_precision_0.6330_recall_0.6244_f1_0.6287_mcc_0.5944_roc0.9123_prc0.6742.pt'
model.load_state_dict(torch.load(model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu'), strict=True)

EMB_PATH = f'./Data/esm2_all'
data_path = '/home/menglingkuan/proj/Data/Phosphoserine_clustered_splited.csv'
data = pd.read_csv(data_path)
train_data = data[data['Set'] == 'train'].head(5)
train_dataset = ProteinDataset(train_data, EMB_PATH, target_aa='S')
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)


model.eval()

all_features = []
all_labels = []


for batch in train_dataloader:
    site_features, full_encoder_output = model.get_site_features(batch)

    for i in range(site_features['features'].shape[0]):
        feature = site_features['features'][i].detach().numpy()  # 转换为numpy数组
        label = site_features['labels'][i].item()  # 获取标签的具体值
        all_features.append(feature)
        all_labels.append(label)

all_features = np.array(all_features)
all_labels = np.array(all_labels)



tsne = TSNE(n_components=2, random_state=42) # 生成2D t-SNE表示
transformed_features = tsne.fit_transform(all_features)

plt.figure(figsize=(8, 8))
# Define custom colors and marker sizes
colors = {1: '#AC3419', 0: 'grey'}  # 1 for positive, 0 for negative
point_size = 5  # Size of points in the plot

# Plotting loop for each unique label
for label in np.unique(all_labels):
    indices = all_labels == label
    plt.scatter(transformed_features[indices, 0], transformed_features[indices, 1],
                color=colors[label],
                label=f"Phosphoserine (S)" if label == 1 else "Non-Phosphoserine (S)",
                s=point_size)  # Use the custom point size

# Enhance the legend
legend = plt.legend(markerscale=2, scatterpoints=1, loc='lower left')   # Increase legend point size
for handle in legend.legendHandles:
    handle.set_sizes([30.0])  # Increase the size of the legend markers

# Formatting and saving the plot
plt.xlabel('t-SNE 1', fontsize=15, weight='bold')
plt.ylabel('t-SNE 2', fontsize=15, weight='bold')
plt.tick_params(axis='both', which='major', labelsize=12)
plt.gca().spines['top'].set_linewidth(2)
plt.gca().spines['right'].set_linewidth(2)
plt.gca().spines['bottom'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)

plt.savefig('tSNE_visualization.svg')
plt.savefig('tSNE_visualization.png')
plt.show()

end_time = time.time()
print(f"Total execution time: {end_time - start_time:.2f} seconds")
