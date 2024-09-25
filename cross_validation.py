import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import KFold
from Data.data_processing import ProteinDataset, collate_fn
from utils import TrainProcessor
from model import TransformerModel
from types import SimpleNamespace
from torch.utils.data import DataLoader
import os

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class CrossValidator:
    def __init__(self, data_path, args):
        self.data = pd.read_csv(data_path)
        self.data = self.data[self.data['Set'] == 'train']
        self.args = args
        self.kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    def run(self):
        fold_metrics = []

        for fold, (train_idx, val_idx) in enumerate(self.kfold.split(self.data)):
            train_samples = self.data.iloc[train_idx]
            val_samples = self.data.iloc[val_idx]

            EMB_PATH = f'./Data/{self.args.emb_type}_all'
            
            train_dataset = ProteinDataset(train_samples, EMB_PATH, target_aa=self.args.target_aa)
            train_dataset, val_dataset = train_dataset.split(val_size=0.1)
            test_dataset = ProteinDataset(val_samples, EMB_PATH, target_aa=self.args.target_aa)
            
            train_dataloader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True, collate_fn=collate_fn)
            val_dataloader = DataLoader(val_dataset, batch_size=self.args.batch_size, shuffle=True, collate_fn=collate_fn)
            test_dataloader = DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False, collate_fn=collate_fn)
            
            model = TransformerModel(emb_size=self.args.emb_size, num_heads=self.args.num_heads, num_layers=self.args.num_layers, hidden_size=self.args.hidden_size, dropout_rate=self.args.dropout_rate, pos_weight=self.args.pos_weight)
            model.to(self.args.device)
            
            train_val_processor = TrainProcessor(model=model, loaders=[train_dataloader, val_dataloader, test_dataloader], args=self.args)
            best_model, test_metrics, training_time = train_val_processor.train()
            
            fold_metrics.append(vars(test_metrics))
            print(f"Fold {fold+1}: Test Loss: {test_metrics.loss:.4f}, Val Accuracy: {test_metrics.acc:.4f}")

            if self.args.save:
                acc = f"{test_metrics.acc:.4f}"
                precision = f"{test_metrics.precision:.4f}"
                recall = f"{test_metrics.recall:.4f}"
                f1 = f"{test_metrics.f1:.4f}"
                mcc = f"{test_metrics.mcc:.4f}"
                auc = f"{test_metrics.auroc:.4f}"
                prc = f"{test_metrics.auprc:.4f}"

                save_dir = f"./cv_results(all)_{self.args.PTM_type}/emb_type_{self.args.emb_type}_epoch{self.args.epochs}b_size{self.args.batch_size}_ods_{self.args.opt_decay_step}wd{self.args.weight_decay}_lr{self.args.lr}_patience{self.args.es_patience}_heads{self.args.num_heads}_hidden{self.args.hidden_size}_dr{self.args.dropout_rate}_pr{self.args.pos_weight}_kernel_size_31"
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                save_path = os.path.join(save_dir, f"{self.args.emb_type}_fold_{fold+1}_acc{acc}_pre{precision}_recall{recall}_f1{f1}_mcc{mcc}_auc{auc}_prc{prc}_train_time{training_time:.2f}s.pt")
                torch.save(best_model.state_dict(), save_path)

        
        # Aggregate and report results across folds
        avg_metrics = {key: np.mean([m[key] for m in fold_metrics]) for key in fold_metrics[0]}
        std_metrics = {key: np.std([m[key] for m in fold_metrics], ddof=1) for key in fold_metrics[0]}

        metrics_str = '_'.join([f"{key}{avg_metrics[key]:.4f}" for key in avg_metrics])
        stats_filename = f"avg:{metrics_str}.txt"
    #    save_dir = f"./cv_results_{self.args.PTM_type}"
    
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        stats_path = os.path.join(save_dir, stats_filename)
        with open(stats_path, 'w') as f:
            f.write("Cross-Validation Results -- Average and Standard Deviation for Metrics across folds:\n")
            for key, value in avg_metrics.items():
                std_metrics_value = std_metrics[key]
                f.write(f"{key.capitalize()}: {value:.4f} ± {std_metrics_value:.4f}\n")

if __name__ == '__main__':
    args = {
        'epochs': 200,
        'batch_size': 32,
        'device': 'cuda:1',
        'opt': 'adam',
        'opt_scheduler': 'step',
        'opt_decay_step': 20,
        'opt_decay_rate': 0.92,
        'weight_decay': 1e-5,
        'lr': 3e-5,
        'es_patience': 20,
        'save': True,
        'emb_size': 1280,
        'num_heads': 8,
        'num_layers': 1,
        'hidden_size': 128,
        'dropout_rate': 0.5,
        'pos_weight': 3,
        'emb_type': 'esm2',
        'PTM_type': 'Phosphothreonine',
        'target_aa':'T'
    }
    args = SimpleNamespace(**args)
    data_path = f'/proj/Data/{args.PTM_type}_clustered_splited.csv'

    validator = CrossValidator(data_path, args)
    validator.run()
   
