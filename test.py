import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from types import SimpleNamespace
from model import TransformerModel
from Data.data_processing import ProteinDataset, collate_fn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, average_precision_score

def test(model, dataloader, device):
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
        binary_pred = np.array([1 if p > 0.5 else 0 for p in pred])

        # Compute metrics
        metrics = {}
        metrics['acc'] = accuracy_score(y, binary_pred)
        metrics['precision'] = precision_score(y, binary_pred)
        metrics['recall'] = recall_score(y, binary_pred)
        metrics['f1'] = f1_score(y, binary_pred)
        metrics['mcc'] = matthews_corrcoef(y, binary_pred)
        metrics['auroc'] = roc_auc_score(y, pred)
        metrics['auprc'] = average_precision_score(y, pred)

        return metrics


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
        PTM_type='Phosphoserine',
        target_aa='S'
    )

    EMB_PATH = f'./Data/{args.emb_type}_all'
    data_path = f'/proj/Data/{args.PTM_type}_clustered_splited.csv'
    data = pd.read_csv(data_path)
    test_data = data[data['Set'] == 'test']
    test_dataset = ProteinDataset(test_data, EMB_PATH, target_aa=args.target_aa)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

    model_path = '/proj/results_Phosphoserine/emb_type_esm2_epochs_200_batch_32_opt_adam_lr_5e-05_patience_40_dropout_0.5_posWeight_3/0_acc0.9384_precision_0.6456_recall_0.6145_f1_0.6297_mcc_0.5963_roc0.9096_prc0.6755.pt'
    model = TransformerModel(emb_size=args.emb_size, num_heads=args.num_heads, num_layers=args.num_layers, hidden_size=args.hidden_size, dropout_rate=args.dropout_rate, pos_weight=args.pos_weight)
    model.to(args.device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    metrics = test(model, test_dataloader, args.device)

    print(f"Test Metrics: Accuracy: {metrics['acc']}, Precision: {metrics['precision']}, Recall: {metrics['recall']}, F1 Score: {metrics['f1']}, MCC: {metrics['mcc']}, AUROC: {metrics['auroc']}, AUPRC: {metrics['auprc']}")
