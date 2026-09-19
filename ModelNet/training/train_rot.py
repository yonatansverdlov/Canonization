import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset

from utils.data import OrderedModelNet40
from utils.models import (
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
    torch.use_deterministic_algorithms(True, warn_only=True)


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


# ====================== CONFIG ====================== #

def cli_overrides(argv):
    """
    Return parser argument names explicitly passed by the user.

    Priority:
        command line > model config > parser defaults
    """
    overrides = set()

    for token in argv:
        if not token.startswith("--"):
            continue

        name = token[2:].split("=")[0]
        name = name.replace("-", "_")
        overrides.add(name)

    return overrides


def apply_model_config(args, explicit_args):
    """
    Load args.config and apply config[args.model].

    Example:
        --model 1
    loads:
        config["1"]

    Command-line arguments override config values.
    """
    if args.model not in ["PurePCA", "FrameAveraging", "Skewness", "RandomFrame"]:
        return args

    config_path = Path(args.config)

    if not config_path.exists():
        raise FileNotFoundError(f"Could not find config file: {config_path}")

    with config_path.open("r") as f:
        config = json.load(f)

    if args.model not in config:
        raise ValueError(
            f"Model '{args.model}' not found in {config_path}. "
            f"Available configs: {list(config.keys())}"
        )

    values = config[args.model]

    for key, value in values.items():
        if key in explicit_args:
            continue

        if key == "model":
            raise ValueError(
                "Do not put 'model' inside the config. "
                "The config block is already selected by --model."
            )

        if not hasattr(args, key):
            raise ValueError(f"Config key '{key}' is not a valid parser argument.")

        setattr(args, key, value)

    return args


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
    train_dataset = make_dataset(
        num_points=args.num_points,
        partition="train",
        ordering=args.ordering,
        use_fps=args.use_fps,
        apply_jitter=args.apply_jitter,
        apply_scale=args.apply_scale,
        apply_rotation=args.apply_rotation,
        apply_random_permutation=args.apply_random_permutation,
    )

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

    train_gen = torch.Generator().manual_seed(args.seed)
    test_gen = torch.Generator().manual_seed(args.seed + 100000)

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
        generator=train_gen,
        **common_loader_kwargs,
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        drop_last=False,
        generator=test_gen,
        **common_loader_kwargs,
    )

    return train_loader, test_loader


# ====================== MODEL ====================== #

def build_model(args, device):
    model_map = {
        "PurePCA": Model1_PurePCA,
        "FrameAveraging": Model2_FrameAveraging,
        "Skewness": Model3_Skewness,
        "RandomFrame": Model4_RandomFrame,
    }

    if args.model not in model_map:
        raise ValueError(
            f"Invalid model '{args.model}'. "
            f"Expected one of {sorted(model_map.keys())}."
        )

    return model_map[args.model]().to(device)


def build_optimizer(args, model):
    if args.optimizer.lower() == "adamw":
        return optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    if args.optimizer.lower() == "sgd" or args.use_sgd:
        return optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )

    return optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )


# ====================== RUN ====================== #

def run_once(args):
    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )

    if args.save_path is None:
        args.save_path = (
            f"checkpoints/{args.model}_{args.exp_name}_seed{args.seed}.pth"
        )

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    train_loader, test_loader = build_loaders(args, device)

    model = build_model(args, device)
    optimizer = build_optimizer(args, model)
    criterion = nn.NLLLoss()

    final_train_acc = float("nan")

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )
        final_train_acc = tr_acc

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:.4f}"
        )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "epoch": args.epochs,
            "train_acc": final_train_acc,
            "model_name": args.model,
        },
        save_path,
    )

    _, final_train_eval_acc = evaluate(
        model,
        train_loader,
        criterion,
        device,
    )
    _, final_test_acc = evaluate(
        model,
        test_loader,
        criterion,
        device,
    )

    gen_gap = final_train_eval_acc - final_test_acc

    return {
        "seed": args.seed,
        "train_acc": final_train_eval_acc,
        "test_acc": final_test_acc,
        "gen_gap": gen_gap,
    }


def run_many_seeds(args):
    all_results = []
    base_exp_name = args.exp_name

    for seed in args.seeds:
        run_args = copy.deepcopy(args)
        run_args.seed = int(seed)
        run_args.exp_name = f"{base_exp_name}_seed{seed}"
        run_args.save_path = None

        result = run_once(run_args)
        all_results.append(result)

    test_accs = np.array(
        [r["test_acc"] for r in all_results],
        dtype=np.float64,
    )
    gen_gaps = np.array(
        [r["gen_gap"] for r in all_results],
        dtype=np.float64,
    )

    ddof = 1 if len(all_results) > 1 else 0

    print()
    print("========== FINAL SUMMARY ==========")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(
        "Test accuracy: "
        f"{test_accs.mean():.6f} ± {test_accs.std(ddof=ddof):.6f}"
    )
    print(
        "Generalization gap (train - test): "
        f"{gen_gaps.mean():.6f} ± {gen_gaps.std(ddof=ddof):.6f}"
    )
    print("===================================")

    return all_results


# ====================== MAIN ====================== #

def main():
    parser = argparse.ArgumentParser(
        description="ModelNet40 rotation/canonicalization experiment"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/PCA.json",
        help="JSON config file. The block is selected by --model.",
    )
    # Core shared defaults
    parser.add_argument("--model", type=str, default="PurePCA", choices=["PurePCA", "FrameAveraging", "Skewness", "RandomFrame"])
    parser.add_argument("--epochs", type=int, default=1600)
    parser.add_argument("--exp_name", type=str, default="PurePCA_SeedStudy_1600")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--test_batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001118604121563404)
    parser.add_argument("--weight_decay", type=float, default=0.000012114201421212455)
    parser.add_argument("--num_points", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=5)
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

    # Dataset / sweep knobs
    parser.add_argument("--dataset", type=str, default="modelnet40")
    parser.add_argument("--ordering", type=str, default="lex")
    parser.add_argument("--dataset_stride", type=int, default=1)
    parser.add_argument("--use_fps", type=str2bool, default=False)
    parser.add_argument("--apply_jitter", type=str2bool, default=False)
    parser.add_argument("--apply_scale", type=str2bool, default=False)
    parser.add_argument("--apply_rotation", type=str2bool, default=True)
    parser.add_argument("--apply_random_permutation", type=str2bool, default=False)

    # Multi-seed
    parser.add_argument("--run_5_seeds", type=str2bool, default=False)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])

    explicit_args = cli_overrides(sys.argv[1:])
    args = parser.parse_args()
    args = apply_model_config(args, explicit_args)

    print("Final args:")
    for k, v in sorted(vars(args).items()):
        print(f"  {k}: {v}")

    if args.eval:
        raise NotImplementedError(
            "Eval-only mode is not implemented in this version. "
            "Training mode is unchanged."
        )

    if args.run_5_seeds:
        run_many_seeds(args)
    else:
        run_once(args)


if __name__ == "__main__":
    main()