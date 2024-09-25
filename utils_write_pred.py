import time
import torch
import torch.optim as optim
import copy
import numpy as np
from torch import nn
from types import SimpleNamespace
from Data.data_processing import data_balance
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, average_precision_score


class TrainProcessor:
    def __init__(self, model, loaders, args):
        self.model = model
        self.train_loader, self.val_loader, self.test_loader = loaders
        print('len train_loader.dataset:', len(self.train_loader.dataset))
        print('len val_loader.dataset:', len(self.val_loader.dataset))
        print('len test_loader.dataset:', len(self.test_loader.dataset))
        self.args = args
        self.optimizer, self.scheduler = self.build_optimizer()

    def build_optimizer(self):
        args = self.args
        filter_fn = filter(lambda p: p.requires_grad, self.model.parameters())
        weight_decay = args.weight_decay
        if args.opt == 'adam':
            optimizer = optim.Adam(filter_fn, lr=args.lr, weight_decay=weight_decay)
        elif args.opt == 'sgd':
            optimizer = optim.SGD(filter_fn, lr=args.lr, momentum=0.95, weight_decay=weight_decay)
        elif args.opt == 'rmsprop':
            optimizer = optim.RMSprop(filter_fn, lr=args.lr, weight_decay=weight_decay)
        elif args.opt == 'adagrad':
            optimizer = optim.Adagrad(filter_fn, lr=args.lr, weight_decay=weight_decay)
        if args.opt_scheduler == 'none':
            return optimizer, None
        elif args.opt_scheduler == 'step':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.opt_decay_step, gamma=args.opt_decay_rate)
        elif args.opt_scheduler == 'reduceOnPlateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                             patience=args.lr_decay_patience,
                                                             factor=args.lr_decay_factor)
        else:
            raise Exception('Unknown optimizer type')

        return optimizer, scheduler

    @torch.no_grad()
    def test(self, model, dataloader):
        model.eval()
        pred_ls = []
        y_ls = []
        for batch in dataloader:
            batch = {k: v.to(self.args.device) for k, v in batch.items()}
            outputs = model(batch)
            preds = outputs.squeeze().cpu().numpy()
            labels = batch['label'].squeeze().cpu().numpy()
            mask = batch['mask'].squeeze().cpu().numpy().astype(bool)

            valid_preds = [pred[mask[i]] for i, pred in enumerate(preds)]
            valid_labels = [label[mask[i]] for i, label in enumerate(labels)]

            pred_ls.extend(valid_preds)
            y_ls.extend(valid_labels)

        balanced_preds, balanced_labels = data_balance(pred_ls, y_ls)
        pred = np.concatenate(pred_ls)
        y = np.concatenate(y_ls)
        binary_pred = np.array([1 if p > 0.5 else 0 for p in pred])

        metrics = {}
        metrics['loss'] = model.weighted_BCEloss(batch, outputs).item()
        metrics['acc'] = accuracy_score(y, binary_pred)
        metrics['precision'] = precision_score(y, binary_pred)
        metrics['recall'] = recall_score(y, binary_pred)
        metrics['f1'] = f1_score(y, binary_pred)
        metrics['mcc'] = matthews_corrcoef(y, binary_pred)
        metrics['auroc'] = roc_auc_score(y, pred)
        metrics['auprc'] = average_precision_score(y, pred)

        return SimpleNamespace(**metrics), y, pred

    def train(self):
        best_val_loss = float('inf')
        best_model = None
        es = 0
        start_time = time.time()
        for epoch in range(self.args.epochs):
            epoch_lr = self.optimizer.param_groups[0]['lr']
            train_epoch_loss = 0.0
            self.model.train()
            for batch in self.train_loader:
                batch = {k: v.to(self.args.device) for k, v in batch.items()}
                self.optimizer.zero_grad()
                outputs = self.model(batch)
                loss = self.model.weighted_BCEloss(batch, outputs)
                loss.backward()
                self.optimizer.step()
                train_epoch_loss += loss.item()

            train_epoch_loss /= len(self.train_loader)
            val_metrics, val_y, val_pred = self.test(self.model, self.val_loader)
            val_epoch_loss, val_epoch_acc, val_epoch_pre, val_epoch_recall, val_epoch_f1, val_epoch_mcc, val_epoch_roc, val_epoch_prc = \
            val_metrics.loss, val_metrics.acc, val_metrics.precision, val_metrics.recall, val_metrics.f1, val_metrics.mcc, val_metrics.auroc, val_metrics.auprc

            self.model.train()
            if self.args.opt_scheduler is None:
                pass
            elif self.args.opt_scheduler == 'reduceOnPlateau':
                self.scheduler.step(val_epoch_loss)
            elif self.args.opt_scheduler == 'step':
                self.scheduler.step()

            log = 'Epoch: {:03d}/{:03d}; ' \
                  'AVG Training Loss (MSE): {:.5f}; ' \
                  'AVG Val Loss (MSE): {:.5f}; ' \
                  'AVG Val Accuracy: {:.5f}; ' \
                  'AVG Val Precision: {:.5f}; ' \
                  'AVG Val Recall: {:.5f}; ' \
                  'AVG Val F1 Score: {:.5f}; ' \
                  'AVG Val MCC: {:.5f}; ' \
                  'AVG Val AUROC: {:.5f}; ' \
                  'AVG Val AUPRC: {:.5f}; ' \
                  'lr: {:8f}'
            print(time.strftime('%H:%M:%S'),
                  log.format(
                      epoch + 1,
                      self.args.epochs,
                      train_epoch_loss,
                      val_epoch_loss,
                      val_epoch_acc,
                      val_epoch_pre,
                      val_epoch_recall,
                      val_epoch_f1,
                      val_epoch_mcc,
                      val_epoch_roc,
                      val_epoch_prc,
                      epoch_lr
                  ))
            if epoch_lr != self.optimizer.param_groups[0]['lr']:
                print('lr has been updated from {:.8f} to {:.8f}'.format(epoch_lr,
                                                                         self.optimizer.param_groups[0]['lr']))
            if val_epoch_loss < best_val_loss:
                best_val_loss = val_epoch_loss
                best_model = copy.deepcopy(self.model)
                es = 0
            else:
                es += 1
                print("Counter {} of patience {}".format(es, self.args.es_patience))
                if es >= self.args.es_patience:
                    print("Early stopping with best_val_loss {:.8f}".format(best_val_loss))
                    break

        training_time = time.time() - start_time
        print(f"Total training time: {training_time:.2f} seconds")
        test_metrics, test_y, test_pred = self.test(best_model, self.test_loader)
        return best_model, test_metrics, training_time, test_y, test_pred
