import argparse
from pathlib import Path
from typing import Dict, List
import torch
from utils.data_funcs import get_dataset, obtain
from utils.models import CNp4CNN


@torch.no_grad()
def rms_distance_batch(x: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    x: (784,)
    Y: (B,784)

    returns: (B,)
    RMS pixel distance.
    """
    diff = Y - x.unsqueeze(0)
    return torch.sqrt(torch.mean(diff * diff, dim=1))


@torch.no_grad()
def flat_to_img(X: torch.Tensor) -> torch.Tensor:
    """
    X: (B,784) -> (B,1,28,28)
    """
    return X.view(X.shape[0], 1, 28, 28)


@torch.no_grad()
def img_to_flat(X: torch.Tensor) -> torch.Tensor:
    """
    X: (B,1,28,28) -> (B,784)
    """
    return X.view(X.shape[0], -1)


@torch.no_grad()
def build_rotated_train_bank(train_images: torch.Tensor) -> torch.Tensor:
    """
    train_images: (N,784)

    returns: (N,4,784)
    """
    X = flat_to_img(train_images)
    rots = torch.stack(
        [torch.rot90(X, k=k, dims=(-2, -1)) for k in range(4)],
        dim=1,
    )  # (N,4,1,28,28)

    return rots.view(train_images.shape[0], 4, -1)


def build_frozen_canonization_model(device: str, seed: int = 0) -> CNp4CNN:
    """
    Builds the random frozen canonizer used by CN(p4 frozen)-CNN.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = CNp4CNN(
        device=device,
        freeze_canonization=True,
    )
    model.eval()
    model.to(device)
    return model


def find_learned_checkpoint() -> str:
    checkpoint_dir = (
        Path(__file__).resolve().parent
        / "data"
        / "trained_models"
        / "learned_can_seed_0_checkpoints"
    )
    checkpoints = sorted(checkpoint_dir.glob("best-learned_can-seed-0*.ckpt"))

    if not checkpoints:
        raise FileNotFoundError(
            "Could not find a trained learned_can checkpoint for seed 0. "
            "Run: python scripts/run_rotated_mnist.py --model learned_can"
        )

    return str(checkpoints[0])


def build_learned_canonization_model(
    device: str,
    checkpoint_path: str | None = None,
) -> CNp4CNN:
    """
    Builds CN(p4)-CNN and loads the trained learned-canonization weights.
    """
    checkpoint_path = checkpoint_path or find_learned_checkpoint()
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    state_dict = checkpoint.get("state_dict", checkpoint)
    model_state = {
        key[len("model."):]: value
        for key, value in state_dict.items()
        if key.startswith("model.")
    }

    if not model_state:
        raise RuntimeError(
            f"No CNp4CNN weights found in checkpoint: {checkpoint_path}"
        )

    model = CNp4CNN(
        device=device,
        freeze_canonization=False,
    )
    model.load_state_dict(model_state, strict=True)
    model.eval()
    model.to(device)

    print("Loaded learned canonizer:", checkpoint_path)
    return model


@torch.no_grad()
def canonize_flat_batch(
    X: torch.Tensor,
    model: CNp4CNN,
    device: str,
) -> torch.Tensor:
    X_img = flat_to_img(X).to(device)
    X_can, _ = model.get_canonized_images(X_img)
    return img_to_flat(X_can).detach().cpu()


@torch.no_grad()
def compute_canon_train_bank(
    train_images: torch.Tensor,
    model: CNp4CNN,
    device: str,
    batch_size: int,
    name: str,
) -> torch.Tensor:
    out = []

    for s in range(0, train_images.shape[0], batch_size):
        X_batch = train_images[s : s + batch_size]
        X_can = canonize_flat_batch(
            X_batch,
            model=model,
            device=device,
        )
        out.append(X_can)

        print(
            f"{name} canonized train "
            f"{min(s + batch_size, train_images.shape[0])}/"
            f"{train_images.shape[0]}"
        )

    return torch.cat(out, dim=0).contiguous()


@torch.no_grad()
def compute_rotated_mnist_nn_scores_from_training_loader(
    train_dataset,
    test_dataset,
    batch_size: int = 4096,
    device: str | None = None,
    reduce_mode: str = "average",
    learned_checkpoint: str | None = None,
) -> Dict[str, float]:
    """
    For each evaluation sample x, computes four coverage distances:

      l2:
          min_y d(x, y)

      group:
          min_y min_{k in {0,1,2,3}} d(x, R^k y)

      can_learned:
          min_y d(c_learned(x), c_learned(y))

      can_frozen:
          min_y d(c_frozen(x), c_frozen(y))
    """
    if reduce_mode not in {"average", "max"}:
        raise ValueError("reduce_mode must be 'average' or 'max'")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    train_images = train_dataset.tensors[0].float().contiguous()
    test_images = test_dataset.tensors[0].float().contiguous()

    train_rot_bank = build_rotated_train_bank(train_images)

    learned_model = build_learned_canonization_model(
        device=device,
        checkpoint_path=learned_checkpoint,
    )
    frozen_model = build_frozen_canonization_model(
        device=device,
        seed=0,
    )

    train_can_learned = compute_canon_train_bank(
        train_images=train_images,
        model=learned_model,
        device=device,
        batch_size=batch_size,
        name="learned",
    )
    train_can_frozen = compute_canon_train_bank(
        train_images=train_images,
        model=frozen_model,
        device=device,
        batch_size=batch_size,
        name="frozen",
    )

    vals_l2: List[float] = []
    vals_group: List[float] = []
    vals_can_learned: List[float] = []
    vals_can_frozen: List[float] = []

    N = train_images.shape[0]

    for i in range(test_images.shape[0]):
        x = test_images[i].to(device)

        x_can_learned = canonize_flat_batch(
            test_images[i : i + 1],
            model=learned_model,
            device=device,
        ).squeeze(0).to(device)

        x_can_frozen = canonize_flat_batch(
            test_images[i : i + 1],
            model=frozen_model,
            device=device,
        ).squeeze(0).to(device)

        best_l2 = None
        best_group = None
        best_can_learned = None
        best_can_frozen = None

        for s in range(0, N, batch_size):
            Y_l2 = train_images[s : s + batch_size].to(device)
            Y_rot = train_rot_bank[s : s + batch_size].to(device)
            Y_can_learned = train_can_learned[s : s + batch_size].to(device)
            Y_can_frozen = train_can_frozen[s : s + batch_size].to(device)

            d_l2 = rms_distance_batch(x, Y_l2)
            bmin_l2 = d_l2.min()

            B = Y_rot.shape[0]
            d_group = rms_distance_batch(
                x,
                Y_rot.view(B * 4, -1),
            ).view(B, 4).min(dim=1).values
            bmin_group = d_group.min()

            d_can_learned = rms_distance_batch(
                x_can_learned,
                Y_can_learned,
            )
            bmin_can_learned = d_can_learned.min()

            d_can_frozen = rms_distance_batch(
                x_can_frozen,
                Y_can_frozen,
            )
            bmin_can_frozen = d_can_frozen.min()

            best_l2 = (
                bmin_l2
                if best_l2 is None
                else torch.minimum(best_l2, bmin_l2)
            )
            best_group = (
                bmin_group
                if best_group is None
                else torch.minimum(best_group, bmin_group)
            )
            best_can_learned = (
                bmin_can_learned
                if best_can_learned is None
                else torch.minimum(best_can_learned, bmin_can_learned)
            )
            best_can_frozen = (
                bmin_can_frozen
                if best_can_frozen is None
                else torch.minimum(best_can_frozen, bmin_can_frozen)
            )

        vals_l2.append(float(best_l2.item()))
        vals_group.append(float(best_group.item()))
        vals_can_learned.append(float(best_can_learned.item()))
        vals_can_frozen.append(float(best_can_frozen.item()))

        if (i + 1) % 100 == 0:
            print(f"processed {i + 1}/{len(test_dataset)}")

    def reduce_vals(vals: List[float]) -> float:
        if reduce_mode == "average":
            return float(sum(vals) / len(vals))
        return float(max(vals))

    return {
        "l2": reduce_vals(vals_l2),
        "group": reduce_vals(vals_group),
        "can_learned": reduce_vals(vals_can_learned),
        "can_frozen": reduce_vals(vals_can_frozen),
    }


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path",
        type=str,
        default=str(Path(__file__).resolve().parent / "data" / "rotated_mnist"),
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=4096,
    )

    parser.add_argument(
        "--reduce_mode",
        type=str,
        default="average",
        choices=["average", "max"],
    )

    parser.add_argument(
        "--split",
        type=str,
        default="valid",
        choices=["valid", "test"],
    )

    parser.add_argument(
        "--learned_checkpoint",
        type=str,
        default=None,
        help="Optional learned_can checkpoint. Defaults to seed 0 best checkpoint.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    obtain(args.data_path)

    train_dataset = get_dataset(args.data_path, split="train")
    eval_dataset = get_dataset(args.data_path, split=args.split)

    results = compute_rotated_mnist_nn_scores_from_training_loader(
        train_dataset=train_dataset,
        test_dataset=eval_dataset,
        batch_size=args.batch_size,
        device=None,
        reduce_mode=args.reduce_mode,
        learned_checkpoint=args.learned_checkpoint,
    )

    print()
print("Results")
print("-" * 36)
print(f"{'method':<20} {'distance':>12}")
print("-" * 32)

for method, value in results.items():
    print(f"{method:<20} {value:>12.6f}")

print("-" * 32)