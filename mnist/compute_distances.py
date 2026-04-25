import argparse
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


def build_blank_learned_canonization_model(device: str) -> CNp4CNN:
    """
    Builds an untrained learned-canonization model.

    No checkpoint is loaded.
    """
    model = CNp4CNN(device=device)
    model.eval()
    model.to(device)
    return model


@torch.no_grad()
def learned_canonize_flat_batch(
    X: torch.Tensor,
    model: CNp4CNN,
    device: str,
) -> torch.Tensor:
    """
    X: (B,784)

    returns: (B,784)

    Applies the blank learned canonization network.
    """
    X_img = flat_to_img(X).to(device)

    X_can, _ = model.get_canonized_images(X_img)

    return img_to_flat(X_can).detach().cpu()


@torch.no_grad()
def compute_learned_canon_train_bank(
    train_images: torch.Tensor,
    model: CNp4CNN,
    device: str,
    batch_size: int,
) -> torch.Tensor:
    """
    Computes learned-canonized versions of all train images.
    """
    out = []

    for s in range(0, train_images.shape[0], batch_size):
        X_batch = train_images[s : s + batch_size]
        X_can = learned_canonize_flat_batch(
            X_batch,
            model=model,
            device=device,
        )
        out.append(X_can)

        print(
            f"learned canonized train "
            f"{min(s + batch_size, train_images.shape[0])}/{train_images.shape[0]}"
        )

    return torch.cat(out, dim=0).contiguous()


@torch.no_grad()
def compute_rotated_mnist_nn_scores_from_training_loader(
    train_dataset,
    test_dataset,
    batch_size: int = 4096,
    device: str | None = None,
    reduce_mode: str = "average",
) -> Dict[str, float]:
    """
    For each test sample x, computes:

      plain:
          min_y d(x, y)

      rot:
          min_y min_{k in {0,1,2,3}} d(x, R^k y)

      learned_can:
          min_y d(c_learned(x), c_learned(y))

    where d is RMS pixel distance.
    """
    if reduce_mode not in {"average", "max"}:
        raise ValueError("reduce_mode must be 'average' or 'max'")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    train_images = train_dataset.tensors[0].float().contiguous()  # (N,784)
    test_images = test_dataset.tensors[0].float().contiguous()    # (M,784)

    train_rot_bank = build_rotated_train_bank(train_images)       # (N,4,784)

    learned_model = build_blank_learned_canonization_model(device)

    train_learned_canon = compute_learned_canon_train_bank(
        train_images=train_images,
        model=learned_model,
        device=device,
        batch_size=batch_size,
    )

    vals_plain: List[float] = []
    vals_rot: List[float] = []
    vals_learned_can: List[float] = []

    N = train_images.shape[0]

    for i in range(test_images.shape[0]):
        x = test_images[i].to(device)

        x_learned_can = learned_canonize_flat_batch(
            test_images[i : i + 1],
            model=learned_model,
            device=device,
        ).squeeze(0).to(device)

        best_plain = None
        best_rot = None
        best_learned_can = None

        for s in range(0, N, batch_size):
            Y_plain = train_images[s : s + batch_size].to(device)
            Y_rot = train_rot_bank[s : s + batch_size].to(device)
            Y_learned_can = train_learned_canon[s : s + batch_size].to(device)

            # 1. plain distance
            d_plain = rms_distance_batch(x, Y_plain)
            bmin_plain = d_plain.min()

            # 2. rotation distance
            B = Y_rot.shape[0]
            d_rot_all = rms_distance_batch(
                x,
                Y_rot.view(B * 4, -1),
            ).view(B, 4)

            d_rot = d_rot_all.min(dim=1).values
            bmin_rot = d_rot.min()

            # 3. learned canonization distance
            d_learned_can = rms_distance_batch(x_learned_can, Y_learned_can)
            bmin_learned_can = d_learned_can.min()

            best_plain = (
                bmin_plain
                if best_plain is None
                else torch.minimum(best_plain, bmin_plain)
            )

            best_rot = (
                bmin_rot
                if best_rot is None
                else torch.minimum(best_rot, bmin_rot)
            )

            best_learned_can = (
                bmin_learned_can
                if best_learned_can is None
                else torch.minimum(best_learned_can, bmin_learned_can)
            )

        vals_plain.append(float(best_plain.item()))
        vals_rot.append(float(best_rot.item()))
        vals_learned_can.append(float(best_learned_can.item()))

        if (i + 1) % 100 == 0:
            print(f"processed {i + 1}/{len(test_dataset)}")

    def reduce_vals(vals: List[float]) -> float:
        if reduce_mode == "average":
            return float(sum(vals) / len(vals))
        return float(max(vals))

    return {
        "l2": reduce_vals(vals_plain),
        "group distance": reduce_vals(vals_rot),
        "canonization": reduce_vals(vals_learned_can),
    }


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path",
        type=str,
        default="data/rotated_mnist",
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
    )

    print()
print("Results")
print("-" * 32)
print(f"{'method':<16} {'distance':>12}")
print("-" * 32)

for method, value in results.items():
    print(f"{method:<16} {value:>12.6f}")

print("-" * 32)