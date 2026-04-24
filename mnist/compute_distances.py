import torch
from typing import Dict, List
from data import get_dataset, obtain, canonize_rot90_lex

@torch.no_grad()
def rms_distance_batch(x: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    x: (784,)
    Y: (B,784)
    returns: (B,)
    distance is in [0,1] if pixels are in [0,1]
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
def canonize_flat_batch(X: torch.Tensor) -> torch.Tensor:
    """
    X: (B,784)
    returns: (B,784)
    """
    X_img = flat_to_img(X)
    X_can = canonize_rot90_lex(X_img)
    return img_to_flat(X_can)


@torch.no_grad()
def build_rotated_train_bank(train_images: torch.Tensor) -> torch.Tensor:
    """
    train_images: (N,784)
    returns: (N,4,784)
    """
    X = flat_to_img(train_images)
    rots = torch.stack(
        [torch.rot90(X, k=k, dims=(-2, -1)) for k in range(4)],
        dim=1
    )  # (N,4,1,28,28)
    return rots.view(train_images.shape[0], 4, -1)


@torch.no_grad()
def compute_rotated_mnist_nn_scores_from_training_loader(
    train_dataset,
    test_dataset,
    batch_size: int = 4096,
    device: str | None = None,
    reduce_mode: str = "average",
) -> Dict[str, float]:
    """
    Uses the same loaded datasets as training:
      train_dataset = get_dataset(data_path, "train")
      test_dataset  = get_dataset(data_path, "test")

    For each test sample x, computes:
      plain: min_y d(x, y)
      rot:   min_y min_{k in {0,1,2,3}} d(x, R^k y)
      canon: min_y d(c(x), c(y))

    where d is RMS pixel distance in [0,1].
    """
    if reduce_mode not in {"average", "max"}:
        raise ValueError("reduce_mode must be 'average' or 'max'")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Exactly the same dataset objects as training gives:
    # TensorDataset(images, labels), with flat images of shape (784,)
    train_images = train_dataset.tensors[0].float().contiguous()  # (N,784)
    test_images = test_dataset.tensors[0].float().contiguous()    # (M,784)

    train_rot_bank = build_rotated_train_bank(train_images)       # (N,4,784)
    train_canon = canonize_flat_batch(train_images)               # (N,784)

    vals_plain: List[float] = []
    vals_rot: List[float] = []
    vals_canon: List[float] = []

    N = train_images.shape[0]

    for i in range(test_images.shape[0]):
        x = test_images[i].to(device)
        x_canon = canonize_flat_batch(test_images[i:i+1]).squeeze(0).to(device)

        best_plain = None
        best_rot = None
        best_canon = None

        for s in range(0, N, batch_size):
            Y_plain = train_images[s:s+batch_size].to(device)     # (B,784)
            Y_rot = train_rot_bank[s:s+batch_size].to(device)     # (B,4,784)
            Y_canon = train_canon[s:s+batch_size].to(device)      # (B,784)

            # 1. plain
            d_plain = rms_distance_batch(x, Y_plain)
            bmin_plain = d_plain.min()

            # 2. best over train rotations
            B = Y_rot.shape[0]
            d_rot_all = rms_distance_batch(x, Y_rot.view(B * 4, -1)).view(B, 4)
            d_rot = d_rot_all.min(dim=1).values
            bmin_rot = d_rot.min()

            # 3. canonized
            d_canon = rms_distance_batch(x_canon, Y_canon)
            bmin_canon = d_canon.min()

            best_plain = bmin_plain if best_plain is None else torch.minimum(best_plain, bmin_plain)
            best_rot = bmin_rot if best_rot is None else torch.minimum(best_rot, bmin_rot)
            best_canon = bmin_canon if best_canon is None else torch.minimum(best_canon, bmin_canon)

        vals_plain.append(float(best_plain.item()))
        vals_rot.append(float(best_rot.item()))
        vals_canon.append(float(best_canon.item()))

        if (i + 1) % 100 == 0:
            print(f"processed {i+1}/{len(test_dataset)}")

    def reduce_vals(vals: List[float]) -> float:
        if reduce_mode == "average":
            return float(sum(vals) / len(vals))
        return float(max(vals))

    return {
        "plain": reduce_vals(vals_plain),
        "rot": reduce_vals(vals_rot),
        "Sort": reduce_vals(vals_canon),
    }



data_path = "/home/yonatans/canonization/mnist/data/rotated_mnist"
obtain(data_path)

train_dataset = get_dataset(data_path, split="train")
test_dataset = get_dataset(data_path, split="test")
val_dataset = get_dataset(data_path, split="valid")
results = compute_rotated_mnist_nn_scores_from_training_loader(
    train_dataset=train_dataset,
    test_dataset=val_dataset,
    batch_size=4096,
    device=None,
    reduce_mode="average",   # or "max"
)

print(results)