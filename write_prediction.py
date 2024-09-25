import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from types import SimpleNamespace
from model import TransformerModel
from Data.data_processing import ProteinDataset, collate_fn

def test_and_save_predictions(model, dataloader, device, output_dir):
    model.eval()
    pred_ls = []
    y_ls = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch)
            preds = outputs.squeeze().cpu().numpy()
            labels = batch['label'].squeeze().cpu().numpy()
            mask = batch['mask'].squeeze().cpu().numpy().astype(bool)

            valid_preds = [pred[mask[i]] for i, pred in enumerate(preds)]
            valid_labels = [label[mask[i]] for i, label in enumerate(labels)]

            pred_ls.extend(valid_preds)
            y_ls.extend(valid_labels)

        pred = np.concatenate(pred_ls)
        y = np.concatenate(y_ls)

        # Save predictions and true labels
        np.save(f'{output_dir}/predictions_{args.emb_type}_{args.PTM_type}.npy', pred)
        np.save(f'{output_dir}/true_labels_{args.emb_type}_{args.PTM_type}.npy', y)

if __name__ == '__main__':
    args = SimpleNamespace(
        batch_size=32,
        device='cuda:1',  
        emb_size=1280,
        num_heads=8,
        num_layers=1,
        hidden_size=128,
        dropout_rate=0.5,
        pos_weight=3,
        emb_type='esm2',
        PTM_type='N6-methyllysine',
        target_aa='K'
    )

    output_dir = '/proj'  
    os.makedirs(output_dir, exist_ok=True)  
    EMB_PATH = f'./Data/{args.emb_type}_all'
    data_path = f'/proj/Data/{args.PTM_type}_clustered_splited.csv'
    data = pd.read_csv(data_path)
    test_data = data[data['Set'] == 'test']
    test_dataset = ProteinDataset(test_data, EMB_PATH, target_aa=args.target_aa)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

    model_path = '/proj/results_N6-methyllysine/emb_type_esm2_epochs_200_batch_32_opt_adam_lr_0.0003_patience_20_dropout_0.5_posWeight_3/4_acc0.9658_precision_0.7742_recall_0.4800_f1_0.5926_mcc_0.5937_roc0.8785_prc0.6131.pt'
    model = TransformerModel(emb_size=args.emb_size, num_heads=args.num_heads, num_layers=args.num_layers, hidden_size=args.hidden_size, dropout_rate=args.dropout_rate, pos_weight=args.pos_weight)
    model.to(args.device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    test_and_save_predictions(model, test_dataloader, args.device, output_dir)
