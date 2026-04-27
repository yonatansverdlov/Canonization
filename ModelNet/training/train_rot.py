import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
import wandb

from data import OrderedModelNet40
from models_rot import (
    Model1_PurePCA,
    Model2_FrameAveraging,
    Model3_Skewness,
    Model4_RandomFrame,
)


# ====================== REPRO ====================== #

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


# ====================== TRAIN / EVAL ====================== #

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for pc, labels in loader:
        pc = pc.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long().view(-1)

        optimizer.zero_grad(set_to_none=True)
        log_probs = model(pc)
        loss = criterion(log_probs, labels)
        loss.backward()
        optimizer.step()

        bs = labels.size(0)
        total_loss += loss.item() * bs
        correct += (log_probs.argmax(dim=1) == labels).sum().item()
        total += bs

    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for pc, labels in loader:
        pc = pc.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long().view(-1)

        log_probs = model(pc)
        loss = criterion(log_probs, labels)

        bs = labels.size(0)
        total_loss += loss.item() * bs
        correct += (log_probs.argmax(dim=1) == labels).sum().item()
        total += bs

    return total_loss / max(total, 1), correct / max(total, 1)


# ====================== DATA ====================== #

def make_dataset(
    num_points,
    partition,
    ordering,
    use_fps,
    apply_jitter,
    apply_scale,
    apply_rotation,
    apply_random_permutation,
):
    return OrderedModelNet40(
        num_points,
        partition=partition,
        ordering=ordering,
        use_fps=use_fps,
        apply_jitter=apply_jitter,
        apply_anisotropic_scale=apply_scale,
        apply_rotation=apply_rotation,
        apply_random_permutation=apply_random_permutation,
    )


def build_loaders(args, device):
    split_gen = torch.Generator().manual_seed(args.seed)

    # Clean base dataset only for splitting
    base_train_dataset = make_dataset(
        num_points=args.num_points,
        partition="train",
        ordering=args.ordering,
        use_fps=args.use_fps,
        apply_jitter=False,
        apply_scale=False,
        apply_rotation=False,
        apply_random_permutation=False,
    )

    val_size = int(len(base_train_dataset) * args.val_split)
    val_size = max(1, val_size)
    train_size = len(base_train_dataset) - val_size

    train_subset, val_subset = random_split(
        base_train_dataset,
        [train_size, val_size],
        generator=split_gen,
    )

    # Augmented train dataset with same indexing
    aug_train_dataset = make_dataset(
        num_points=args.num_points,
        partition="train",
        ordering=args.ordering,
        use_fps=args.use_fps,
        apply_jitter=args.apply_jitter,
        apply_scale=args.apply_scale,
        apply_rotation=args.apply_rotation,
        apply_random_permutation=args.apply_random_permutation,
    )

    train_dataset = Subset(aug_train_dataset, train_subset.indices)
    val_dataset = Subset(base_train_dataset, val_subset.indices)

    test_dataset = make_dataset(
        num_points=args.num_points,
        partition="test",
        ordering=args.ordering,
        use_fps=args.use_fps,
        apply_jitter=False,
        apply_scale=False,
        apply_rotation=args.apply_rotation,
        apply_random_permutation=False,
    )

    common_loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=worker_init_fn if args.num_workers > 0 else None,
    )

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=False,
        **common_loader_kwargs,
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        drop_last=False,
        **common_loader_kwargs,
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        drop_last=False,
        **common_loader_kwargs,
    )

    return train_loader, val_loader, test_loader


# ====================== MAIN ====================== #

# ====================== MAIN ====================== #

def main():
    parser = argparse.ArgumentParser(description="ModelNet40 rotation/canonicalization experiment")

    # Core
    parser.add_argument("--model", type=str, default="global_mlp")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--test_batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.0009614328324244756)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_points", type=int, default=1024)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--no_cuda", type=str2bool, default=False)
    parser.add_argument("--eval", type=str2bool, default=False)

    # Optimization & Architecture Knobs
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--use_sgd", type=str2bool, default=False)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--num_bands", type=int, default=3)
    parser.add_argument("--fourier_scale", type=float, default=0.1)
    parser.add_argument("--label_smoothing", type=float, default=0.2)
    parser.add_argument("--trans_dim", type=int, default=216)
    parser.add_argument("--trans_depth", type=int, default=4)
    parser.add_argument("--trans_heads", type=int, default=6)
    parser.add_argument("--drop_path_rate", type=float, default=0.1)

    # W&B
    parser.add_argument("--wandb_project", type=str, default="modelnet40-canonization")
    parser.add_argument("--disable_wandb", action="store_true")
    parser.add_argument("--exp_name", type=str, default="Best_MLP_Hilbert")

    # Dataset / sweep knobs
    parser.add_argument("--dataset", type=str, default="modelnet40")
    parser.add_argument("--ordering", type=str, default="hilbert")
    parser.add_argument("--dataset_stride", type=int, default=1)
    parser.add_argument("--use_fps", type=str2bool, default=False)
    parser.add_argument("--apply_jitter", type=str2bool, default=False)
    parser.add_argument("--apply_scale", type=str2bool, default=False)
    parser.add_argument("--apply_rotation", type=str2bool, default=False)
    parser.add_argument("--apply_random_permutation", type=str2bool, default=False)

    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    model_map = {
        "1": Model1_PurePCA,
        "2": Model2_FrameAveraging,
        "3": Model3_Skewness,
        "4": Model4_RandomFrame,
        # "global_mlp": GlobalMLP,  # Uncomment and map this to your actual MLP class
    }

    if args.model not in model_map and args.model != "global_mlp":
        raise ValueError(f"Invalid model '{args.model}'. Expected one of {sorted(model_map.keys())}.")

    use_wandb = not args.disable_wandb
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.exp_name,
            config=vars(args),
        )
        wandb.define_metric("epoch")
        wandb.define_metric("train/loss", step_metric="epoch")
        wandb.define_metric("train/acc", step_metric="epoch")
        wandb.define_metric("val/loss", step_metric="epoch")
        wandb.define_metric("val/acc", step_metric="epoch")
        wandb.define_metric("val/best_acc", step_metric="epoch")
        wandb.define_metric("test/loss", step_metric="epoch")
        wandb.define_metric("test/acc", step_metric="epoch")
        wandb.define_metric("test/best_acc", step_metric="epoch")

    if args.save_path is None:
        run_suffix = f"_{wandb.run.id}" if use_wandb else ""
        args.save_path = f"best_model_m{args.model}_{args.exp_name}{run_suffix}.pth"

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader = build_loaders(args, device)

    # Initialize model (Make sure to hook up global_mlp correctly!)
    if args.model == "global_mlp":
        print("WARNING: 'global_mlp' chosen. Make sure it is correctly imported and initialized.")
        # model = GlobalMLP(...).to(device)
        model = None  # Replace this with your MLP init
    else:
        model = model_map[args.model]().to(device)

    # Note: Added logic to use the requested optimizer setting
    if args.optimizer.lower() == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == "sgd" or args.use_sgd:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    criterion = nn.NLLLoss()

    best_val_acc = -1.0
    best_test_acc = -1.0
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        # Save by TEST accuracy, as requested
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "epoch": epoch,
                    "train_acc": tr_acc,
                    "val_acc": val_acc,
                    "test_acc": test_acc,
                    "best_test_acc": best_test_acc,
                    "best_val_acc": best_val_acc,
                    "best_epoch": best_epoch,
                    "model_name": args.model,
                },
                save_path,
            )

        if use_wandb:
            wandb.log(
                {
                    "epoch": epoch,
                    "train/loss": tr_loss,
                    "train/acc": tr_acc,
                    "val/loss": val_loss,
                    "val/acc": val_acc,
                    "val/best_acc": best_val_acc,
                    "test/loss": test_loss,
                    "test/acc": test_acc,
                    "test/best_acc": best_test_acc,
                }
            )

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | "
            f"Best Test Acc: {best_test_acc:.4f}"
        )

    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    final_test_loss, final_test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nBest checkpoint: epoch {best_epoch:03d} | Best Test Acc: {best_test_acc:.4f}")
    print(f"Reloaded checkpoint -> Final Test Loss: {final_test_loss:.4f} | Final Test Acc: {final_test_acc:.4f}")

    if use_wandb:
        wandb.log(
            {
                "final_test/loss": final_test_loss,
                "final_test/acc": final_test_acc,
                "final_test/best_acc": best_test_acc,
                "final_test/best_epoch": best_epoch,
            }
        )
        wandb.finish()


if __name__ == "__main__":
    main()