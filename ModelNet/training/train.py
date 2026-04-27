#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import sys
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import numpy as np
import sklearn.metrics as metrics

from data import OrderedModelNet40
from models import GlobalMLPClassifier, PointTransformerClassifier
from util import IOStream
import wandb


# --- NEW: Bulletproof Boolean Parser for WandB Sweeps ---
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def _init_(args):
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    if not os.path.exists('checkpoints/' + args.exp_name + '/models'):
        os.makedirs('checkpoints/' + args.exp_name + '/models')

    os.system('cp train.py checkpoints/' + args.exp_name + '/train.py.backup')
    os.system('cp models.py checkpoints/' + args.exp_name + '/models.py.backup')
    os.system('cp data.py checkpoints/' + args.exp_name + '/data.py.backup')
    os.system('cp util.py checkpoints/' + args.exp_name + '/util.py.backup')


def train(args, io):
    # --- Updated Train DataLoader ---
    train_loader = DataLoader(
        OrderedModelNet40(
            partition='train',
            num_points=args.num_points,
            ordering=args.ordering,
            dataset_stride=args.dataset_stride,
            use_fps=args.use_fps,
            apply_jitter=args.apply_jitter,
            apply_anisotropic_scale=args.apply_scale,
            apply_random_permutation=args.apply_random_permutation,
            apply_rotation=args.apply_rotation  # <-- NEW
        ),
        num_workers=8, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # --- Updated Test DataLoader ---
    test_loader = DataLoader(
        OrderedModelNet40(
            partition='test',
            num_points=args.num_points,
            ordering=args.ordering,
            dataset_stride=1,
            use_fps=args.use_fps,
            apply_jitter=False,  # Enforce safety
            apply_anisotropic_scale=False,  # Enforce safety
            apply_random_permutation=False,  # Enforce safety
            apply_rotation=False  # <-- NEW (Enforce safety)
        ),
        num_workers=8, batch_size=args.test_batch_size, shuffle=False, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    # --- Model Instantiation ---
    if args.model == 'global_mlp':
        model = GlobalMLPClassifier(
            num_classes=40,
            num_points=args.num_points,
            num_bands=args.num_bands,
            fourier_scale=args.fourier_scale,
            dropout=args.dropout,
            ordering_type=args.ordering
        ).to(device)
    elif args.model == 'point_transformer':
        model = PointTransformerClassifier(
            num_classes=40,
            dim=args.trans_dim,
            depth=args.trans_depth,
            heads=args.trans_heads,
            drop_rate=args.dropout,
            drop_path_rate=args.drop_path_rate
        ).to(device)
    else:
        raise Exception(f"Model {args.model} not implemented")

    model = nn.DataParallel(model)

    print("Let's use", torch.cuda.device_count(), "GPUs!")

    opt_choice = 'sgd' if args.use_sgd else args.optimizer.lower()

    if opt_choice == 'sgd':
        print("Using SGD")
        start_lr = args.lr * 100
        opt = optim.SGD(model.parameters(), lr=start_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif opt_choice == 'adamw':
        print("Using AdamW")
        start_lr = args.lr
        opt = optim.AdamW(model.parameters(), lr=start_lr, weight_decay=args.weight_decay)
    else:
        print("Using Adam")
        start_lr = args.lr
        opt = optim.Adam(model.parameters(), lr=start_lr, weight_decay=args.weight_decay)

    min_lr = start_lr * 0.001
    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=min_lr)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    best_test_acc = 0
    global_step = 0  # <--- Initialize Global Step Tracker

    for epoch in range(args.epochs):
        # --- Train ---
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
        for data, label in train_bar:
            data, label = data.to(device), label.to(device).squeeze()

            batch_size = data.size()[0]

            opt.zero_grad()
            logits = model(data)
            loss = criterion(logits, label)

            if torch.isnan(loss):
                print(f"\nFATAL ERROR: NaN loss detected at Epoch {epoch}!")
                wandb.run.summary["status"] = "failed_due_to_nan"
                sys.exit(1)

            loss.backward()
            opt.step()

            global_step += 1  # <--- Increment Global Step Tracker

            # <--- Log Step-Level Metrics
            wandb.log({
                "train/step_loss": loss.item(),
                "global_step": global_step
            })

            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())

        scheduler.step()
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)

        train_acc = metrics.accuracy_score(train_true, train_pred)
        train_avg_acc = metrics.balanced_accuracy_score(train_true, train_pred)

        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (
            epoch, train_loss * 1.0 / count, train_acc, train_avg_acc)
        io.cprint(outstr)

        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss * 1.0 / count,
            "train/acc": train_acc,
            "train/avg_acc": train_avg_acc,
            "global_step": global_step
        })

        # --- Test ---
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []

        with torch.no_grad():
            for data, label in test_loader:
                data, label = data.to(device), label.to(device).squeeze()
                batch_size = data.size()[0]

                logits = model(data)
                loss = criterion(logits, label)
                preds = logits.max(dim=1)[1]

                count += batch_size
                test_loss += loss.item() * batch_size
                test_true.append(label.cpu().numpy())
                test_pred.append(preds.detach().cpu().numpy())

        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)

        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)

        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (
            epoch, test_loss * 1.0 / count, test_acc, avg_per_class_acc)
        io.cprint(outstr)

        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), 'checkpoints/%s/models/model.pt' % args.exp_name)

        wandb.log({
            "epoch": epoch,
            "test/loss": test_loss * 1.0 / count,
            "test/acc": test_acc,
            "test/avg_acc": avg_per_class_acc,
            "test/best_acc": best_test_acc,
            "lr": opt.param_groups[0]['lr']
        })


def test(args, io):
    # --- Updated Eval DataLoader ---
    test_loader = DataLoader(
        OrderedModelNet40(
            partition='test',
            num_points=args.num_points,
            ordering=args.ordering,
            dataset_stride=1,
            use_fps=args.use_fps,
            apply_jitter=False,
            apply_anisotropic_scale=False,
            apply_random_permutation=False,
            apply_rotation=False  # <-- NEW (Enforce safety)
        ),
        num_workers=8, batch_size=args.test_batch_size, shuffle=False, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    if args.model == 'global_mlp':
        model = GlobalMLPClassifier(
            num_classes=40,
            num_points=args.num_points,
            num_bands=args.num_bands,
            fourier_scale=args.fourier_scale,
            dropout=args.dropout,
            ordering_type=args.ordering
        ).to(device)
    elif args.model == 'point_transformer':
        model = PointTransformerClassifier(
            num_classes=40,
            dim=args.trans_dim,
            depth=args.trans_depth,
            heads=args.trans_heads,
            drop_rate=args.dropout,
            drop_path_rate=args.drop_path_rate
        ).to(device)
    else:
        raise Exception(f"Model {args.model} not implemented")

    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.eval()

    test_pred = []
    test_true = []

    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.to(device), label.to(device).squeeze()
            logits = model(data)
            preds = logits.max(dim=1)[1]
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())

    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)

    outstr = 'Test :: test acc: %.6f, test avg acc: %.6f' % (test_acc, avg_per_class_acc)
    io.cprint(outstr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pre-ordered Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='Best_MLP_Hilbert', metavar='N', help='Name of the experiment')

    parser.add_argument('--ordering', type=str, default='hilbert', choices=['lex', 'hilbert', 'ply', 'pca'],
                        help='Which canonical dataset to load')
    parser.add_argument('--model', type=str, default='global_mlp', choices=['global_mlp', 'point_transformer'],
                        help='Model to use')

    # --- Data Processing & Augmentation Args (REVISED FOR WANDB) ---
    parser.add_argument('--dataset_stride', type=int, default=1,
                        help='Subsample the dataset by taking every Nth pointcloud')
    parser.add_argument('--use_fps', type=str2bool, nargs='?', const=True, default=False,
                        help='Use Farthest Point Sampling instead of stride-based downsampling')
    parser.add_argument('--apply_jitter', type=str2bool, nargs='?', const=True, default=False,
                        help='Apply random jitter augmentation to train data')
    parser.add_argument('--apply_scale', type=str2bool, nargs='?', const=True, default=False,
                        help='Apply anisotropic scaling augmentation to train data')
    parser.add_argument('--apply_random_permutation', type=str2bool, nargs='?', const=True, default=False,
                        help='Apply random permutation to point ordering during training')

    # --- NEW: Added random rotation toggle ---
    parser.add_argument('--apply_rotation', type=str2bool, nargs='?', const=True, default=False,
                        help='Apply random rotation augmentation to train data')

    # --- Transformer Hyperparameters ---
    parser.add_argument('--trans_dim', type=int, default=216, help='Transformer embedding dimension')
    parser.add_argument('--trans_depth', type=int, default=4, help='Number of Transformer layers')
    parser.add_argument('--trans_heads', type=int, default=6, help='Number of attention heads')
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help='Stochastic depth rate')

    # --- Global MLP Hyperparameters ---
    parser.add_argument('--num_bands', type=int, default=3, help='Number of Fourier bands for Global MLP')
    parser.add_argument('--fourier_scale', type=float, default=0.1,
                        help='Scale for Random Fourier Features')

    # --- Standard Hyperparameters ---
    parser.add_argument('--batch_size', type=int, default=256, metavar='batch_size', help='Size of batch')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size', help='Size of batch')
    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of episodes to train')
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw', 'sgd'],
                        help='Optimizer to use')
    parser.add_argument('--use_sgd', type=str2bool, nargs='?', const=True, default=False,
                        help='Use SGD (Default is Adam)')
    parser.add_argument('--lr', type=float, default=0.0009614328324244756, metavar='LR', help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for the optimizer')
    parser.add_argument('--label_smoothing', type=float, default=0.2,
                        help='Label smoothing epsilon for cross entropy loss')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')

    # --- System ---
    parser.add_argument('--dataset', type=str, default='modelnet40', choices=['modelnet40'])
    parser.add_argument('--num_points', type=int, default=1024, help='num of points to use')
    parser.add_argument('--eval', type=str2bool, nargs='?', const=True, default=False, help='evaluate the model')
    parser.add_argument('--no_cuda', type=str2bool, nargs='?', const=True, default=False, help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=4, metavar='S', help='random seed (default: 4)')
    parser.add_argument('--model_path', type=str, default='', metavar='N', help='Pretrained model path')

    args = parser.parse_args()

    wandb.init(project="canon", name=args.exp_name, config=vars(args))

    # <--- DEFINED METRICS FOR BOTH EPOCH AND STEP LOGGING --->
    wandb.define_metric("epoch")
    wandb.define_metric("global_step")
    wandb.define_metric("train/step_loss", step_metric="global_step")
    wandb.define_metric("train/loss", step_metric="epoch")
    wandb.define_metric("train/acc", step_metric="epoch")
    wandb.define_metric("train/avg_acc", step_metric="epoch")
    wandb.define_metric("test/*", step_metric="epoch")
    wandb.define_metric("lr", step_metric="epoch")

    _init_(args)

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)

    if args.cuda:
        io.cprint(f'Using GPU: {torch.cuda.current_device()} from {torch.cuda.device_count()} devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        test(args, io)