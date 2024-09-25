import pandas as pd
import torch
from types import SimpleNamespace
import numpy as np
import random
from Data.data_processing import ProteinDataset, collate_fn
from utils import TrainProcessor
from torch.utils.data import DataLoader
from model import TransformerModel
import os


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)  
torch.backends.cudnn.deterministic = True  
torch.backends.cudnn.benchmark = False

if __name__ == '__main__':

    args = {
        'epochs': 200,
        'batch_size': 32,
        'device': 'cuda:1',
        'opt': 'adam',
        'opt_scheduler': 'step',
        'opt_decay_step': 20,
        'opt_decay_rate': 0.92,
        'weight_decay': 1e-4,
        'lr': 5e-5,
        'es_patience': 40,
        'save': True,
        'emb_size': 1280,
        'num_heads': 8,
        'num_layers': 1,
        'hidden_size': 128,
        'dropout_rate': 0.5,
        'pos_weight': 3,
        'emb_type': 'esm2',
        'PTM_type': 'N6-succinyllysine',
        'target_aa':'K'
    }
    args = SimpleNamespace(**args)
    print(args)
    
    EMB_PATH = f'./Data/{args.emb_type}_all'
    data_path = f'/home/menglingkuan/proj/Data/{args.PTM_type}_clustered_splited.csv'
    
    data = pd.read_csv(data_path)


    train_data = data[data['Set'] == 'train']
    test_data = data[data['Set'] == 'test']
   # print(len(test_data),test_data)
   

    train_dataset = ProteinDataset(train_data, EMB_PATH,target_aa=args.target_aa)
    train_dataset, val_dataset = train_dataset.split(val_size=0.1)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
                                
    test_dataset = ProteinDataset(test_data, EMB_PATH,target_aa=args.target_aa)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
 


    metrics_list = []

    for i in range(5):
        model = TransformerModel(emb_size=args.emb_size, num_heads=args.num_heads, num_layers=args.num_layers, hidden_size=args.hidden_size, dropout_rate=args.dropout_rate, pos_weight=args.pos_weight)
        model.to(args.device)
        
        train_val = TrainProcessor(model=model, loaders=[train_dataloader, val_dataloader, test_dataloader], args=args)
        best_model, test_metrics, training_time = train_val.train()
        
        metrics_list.append(vars(test_metrics))
        
        print('test loss: {:4f}; test acc: {:4f}; test precision: {:4f}; test recall: {:4f}; test f1: {:4f}; test mcc: {:4f}; test auroc: {:4f}; test auprc: {:.4f}'.format(
            test_metrics.loss, test_metrics.acc, test_metrics.precision, test_metrics.recall, test_metrics.f1, test_metrics.mcc, test_metrics.auroc, test_metrics.auprc))

        if args.save:
            save_dir = f"./results_{args.PTM_type}/emb_type_{args.emb_type}_epochs_{args.epochs}_batch_{args.batch_size}_opt_{args.opt}_lr_{args.lr}_patience_{args.es_patience}_dropout_{args.dropout_rate}_posWeight_{args.pos_weight}"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_path = os.path.join(save_dir, '{}_acc{:.4f}_precision_{:.4f}_recall_{:.4f}_f1_{:.4f}_mcc_{:.4f}_roc{:.4f}_prc{:.4f}.pt'.format(i, test_metrics.acc, test_metrics.precision, test_metrics.recall, test_metrics.f1, test_metrics.mcc, test_metrics.auroc, test_metrics.auprc))
            torch.save(best_model.state_dict(), save_path)

    # Calculate the average and standard deviation of all metrics
    avg_metrics = {key: np.mean([m[key] for m in metrics_list]) for key in metrics_list[0]}
    std_metrics = {key: np.std([m[key] for m in metrics_list], ddof=1) for key in metrics_list[0]}  # Using ddof=1 for sample standard deviation

    # Save average metrics with standard deviation
    avg_std_save_path = os.path.join(save_dir, '{}, average_std_acc{:.4f}_{:.4f}_precision_{:.4f}_{:.4f}_recall_{:.4f}_{:.4f}_f1_{:.4f}_{:.4f}_mcc_{:.4f}_{:.4f}_roc{:.4f}_{:.4f}_prc{:.4f}_{:.4f}.txt'.format(
        args.emb_type,
        avg_metrics['acc'], std_metrics['acc'],
        avg_metrics['precision'], std_metrics['precision'],
        avg_metrics['recall'], std_metrics['recall'],
        avg_metrics['f1'], std_metrics['f1'],
        avg_metrics['mcc'], std_metrics['mcc'],
        avg_metrics['auroc'], std_metrics['auroc'],
        avg_metrics['auprc'], std_metrics['auprc']))
    with open(avg_std_save_path, 'w') as f:
        f.write('Average and SD Metrics: ' + str({k: (v, std_metrics[k]) for k, v in avg_metrics.items()}))

    
    print('Average and Standard Deviation for Metrics:')
    print('{}, Acc: {:.4f} ± {:.4f}, Precision: {:.4f} ± {:.4f}, Recall: {:.4f} ± {:.4f}, F1: {:.4f} ± {:.4f}, MCC: {:.4f} ± {:.4f}, AUROC: {:.4f} ± {:.4f}, AUPRC: {:.4f} ± {:.4f}'.format(
    args.emb_type,
    avg_metrics['acc'], std_metrics['acc'],
    avg_metrics['precision'], std_metrics['precision'],
    avg_metrics['recall'], std_metrics['recall'],
    avg_metrics['f1'], std_metrics['f1'],
    avg_metrics['mcc'], std_metrics['mcc'],
    avg_metrics['auroc'], std_metrics['auroc'],
    avg_metrics['auprc'], std_metrics['auprc']))
