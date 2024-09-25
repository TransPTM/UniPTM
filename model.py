import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch


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
        emb = emb.transpose(1, 2)  
        emb = self.cnn(emb)
        emb = emb.transpose(1, 2)  
        x = self.encoder(emb)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)

        return torch.sigmoid(x)
    
     

    def weighted_BCEloss(self, batch, outputs):
        mask = batch['mask'].squeeze(0).bool()
        true_y = batch['label'].squeeze(0)[mask].float()
        pred_y = outputs.squeeze(0)[mask].squeeze(-1)
        weights = torch.ones_like(true_y)  
        if self.pos_weight is not None:
            weights[true_y == 1] = self.pos_weight  
        loss = F.binary_cross_entropy(pred_y, true_y, weight=weights)
        return loss

    def BCEloss(self, batch, outputs):
        mask = batch['mask']
        mask = mask.squeeze(0).bool()
        true_y = batch['label'].squeeze(0)[mask].float()
        pred_y = outputs.squeeze(0)[mask].squeeze(-1)
        loss = F.binary_cross_entropy(pred_y, true_y)
        return loss
