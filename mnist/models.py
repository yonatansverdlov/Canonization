import math

import kornia as K
import torch
import torch.nn as nn
import torch.nn.functional as F

class RotationEquivariantConvLift(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        num_rotations=4,
        stride=1,
        padding=0,
        bias=True,
        device="cuda",
    ):
        super().__init__()
        self.weights = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size, device=device)
        )
        torch.nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels, device=device))
            torch.nn.init.zeros_(self.bias)
        else:
            self.bias = None

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.num_rotations = num_rotations
        self.kernel_size = kernel_size

    def get_rotated_weights(self, weights):
        device = weights.device
        weights = weights.flatten(0, 1).unsqueeze(0).repeat(self.num_rotations, 1, 1, 1)
        rotated_weights = K.geometry.rotate(
            weights,
            torch.linspace(0.0, 360.0, steps=self.num_rotations + 1, dtype=torch.float32, device=device)[
                : self.num_rotations
            ],
        )
        rotated_weights = rotated_weights.reshape(
            self.num_rotations,
            self.out_channels,
            self.in_channels,
            self.kernel_size,
            self.kernel_size,
        ).transpose(0, 1)
        return rotated_weights.flatten(0, 1)

    def forward(self, x):
        batch_size = x.shape[0]
        rotated_weights = self.get_rotated_weights(self.weights)
        x = F.conv2d(x, rotated_weights, stride=self.stride, padding=self.padding)
        x = x.reshape(batch_size, self.out_channels, self.num_rotations, x.shape[2], x.shape[3])
        if self.bias is not None:
            x = x + self.bias[None, :, None, None, None]
        return x


class RotationEquivariantConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        num_rotations=4,
        stride=1,
        padding=0,
        bias=True,
        device="cuda",
    ):
        super().__init__()
        self.weights = nn.Parameter(
            torch.empty(out_channels, in_channels, num_rotations, kernel_size, kernel_size, device=device)
        )
        torch.nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels, device=device))
            torch.nn.init.zeros_(self.bias)
        else:
            self.bias = None

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.num_rotations = num_rotations
        self.kernel_size = kernel_size

        indices = torch.arange(num_rotations, device=device).view(1, 1, num_rotations, 1, 1).repeat(
            num_rotations, out_channels * in_channels, 1, kernel_size, kernel_size
        )
        self.permute_indices_along_group = (
            (indices - torch.arange(num_rotations, device=device)[:, None, None, None, None]) % num_rotations
        )
        self.angle_list = torch.linspace(
            0.0, 360.0, steps=num_rotations + 1, dtype=torch.float32, device=device
        )[:num_rotations]

    def get_rotated_permuted_weights(self, weights):
        weights = weights.flatten(0, 1).unsqueeze(0).repeat(self.num_rotations, 1, 1, 1, 1)
        permuted_weights = torch.gather(weights, 2, self.permute_indices_along_group)
        rotated_permuted_weights = K.geometry.rotate(
            permuted_weights.flatten(1, 2),
            self.angle_list,
        )
        rotated_permuted_weights = rotated_permuted_weights.reshape(
            self.num_rotations,
            self.out_channels,
            self.in_channels,
            self.num_rotations,
            self.kernel_size,
            self.kernel_size,
        ).transpose(0, 1).reshape(
            self.out_channels * self.num_rotations,
            self.in_channels * self.num_rotations,
            self.kernel_size,
            self.kernel_size,
        )
        return rotated_permuted_weights

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.flatten(1, 2)
        rotated_permuted_weights = self.get_rotated_permuted_weights(self.weights)
        x = F.conv2d(x, rotated_permuted_weights, stride=self.stride, padding=self.padding)
        x = x.reshape(batch_size, self.out_channels, self.num_rotations, x.shape[2], x.shape[3])
        if self.bias is not None:
            x = x + self.bias[None, :, None, None, None]
        return x


class CNN(nn.Module):
    def __init__(self, in_shape=(1, 28, 28), out_channels=32, num_layers=6):
        super().__init__()
        encoder_layers = []
        self.im_shape = (1, 28, 28)
        for i in range(num_layers):
            if i == 0:
                encoder_layers.append(nn.Conv2d(in_shape[0], out_channels, 3, 1))
            elif i % 3 == 2:
                encoder_layers.append(nn.Conv2d(out_channels, 2 * out_channels, 5, 2, 1))
                out_channels *= 2
            else:
                encoder_layers.append(nn.Conv2d(out_channels, out_channels, 3, 1))

            encoder_layers.append(nn.BatchNorm2d(out_channels))
            encoder_layers.append(nn.ReLU())

            if i % 3 == 2:
                encoder_layers.append(nn.Dropout2d(0.4))

        self.encoder = nn.Sequential(*encoder_layers)
        
        with torch.no_grad():
            dummy = torch.zeros(1, *self.im_shape)
            out = self.encoder(dummy)
            flat_dim = out.shape[1] * out.shape[2] * out.shape[3]

        self.predictor = nn.Linear(flat_dim, 10)

    def forward(self, x):
        feats = self.encoder(x)
        feats = feats.view(x.shape[0], -1)
        return self.predictor(feats)




class CanonizationNetwork(nn.Module):
    def __init__(
        self,
        in_shape,
        out_channels,
        kernel_size,
        num_rotations=4,
        num_layers=3,
        device="cuda",
    ):
        super().__init__()

        layer_list = [
            RotationEquivariantConvLift(
                in_shape[0],
                out_channels,
                kernel_size,
                num_rotations,
                device=device,
            )
        ]
        for _ in range(num_layers - 1):
            layer_list.append(nn.ReLU())
            layer_list.append(
                RotationEquivariantConv(
                    out_channels,
                    out_channels,
                    1,
                    num_rotations,
                    device=device,
                )
            )
        self.eqv_network = nn.Sequential(*layer_list)

    def forward(self, x):
        # output shape before reduction: (B, C, 4, H, W)
        feature_map = self.eqv_network(x)
        # reduce over channels and spatial dimensions -> (B, 4)
        feature_fibres = torch.mean(feature_map, dim=(1, 3, 4))
        return feature_fibres


class EquivariantCanonizationNetwork(nn.Module):
    def __init__(
        self,
        base_encoder,
        in_shape,
        num_classes,
        canonization_out_channels=16,
        canonization_num_layers=3,
        canonization_kernel_size=3,
        canonization_beta=1.0,
        num_rotations=4,
        device="cuda",
        batch_size=256,
    ):
        super().__init__()

        self.canonization_network = CanonizationNetwork(
            in_shape=in_shape,
            out_channels=canonization_out_channels,
            kernel_size=canonization_kernel_size,
            num_rotations=num_rotations,
            num_layers=canonization_num_layers,
            device=device,
        )
        self.freeze = True
        self.base_encoder = base_encoder
        self.num_rotations = num_rotations
        self.beta = canonization_beta

        with torch.no_grad():
            dummy = torch.zeros(batch_size, *in_shape, device=device)
            out_shape = self.base_encoder(dummy).shape

        if len(out_shape) == 4:
            flat_dim = out_shape[1] * out_shape[2] * out_shape[3]
        elif len(out_shape) == 2:
            flat_dim = out_shape[1]
        else:
            raise ValueError("Base encoder output shape must be 2D or 4D.")

        self.predictor = nn.Linear(flat_dim, num_classes)

    def fibres_to_group(self, fibre_activations):
        device = fibre_activations.device

        fibre_activations_one_hot = torch.nn.functional.one_hot(
            torch.argmax(fibre_activations, dim=-1),
            self.num_rotations,
        ).float()

        fibre_activations_soft = torch.nn.functional.softmax(
            self.beta * fibre_activations,
            dim=-1,
        )

        angles = torch.linspace(
            0.0, 360.0, self.num_rotations + 1, device=device
        )[:self.num_rotations]

        if self.training:
            angles = torch.sum(
                (
                    fibre_activations_one_hot
                    + fibre_activations_soft
                    - fibre_activations_soft.detach()
                )
                * angles,
                dim=-1,
            )
        else:
            angles = torch.sum(fibre_activations_one_hot * angles, dim=-1)

        return angles

    def inverse_action(self, x, fibre_activations):
        angles = self.fibres_to_group(fibre_activations)
        x = K.geometry.rotate(x, -angles)
        return x, angles

    def get_canonized_images(self, x):
        fibre_activations = self.canonization_network(x)
        x_canonized, group = self.inverse_action(x, fibre_activations)
        return x_canonized, group

    def forward(self, x):
        batch_size = x.shape[0]
        if self.freeze:
            with torch.no_grad():
                x_canonized, _ = self.get_canonized_images(x)
        else:
            x_canonized, _ = self.get_canonized_images(x)
        reps = self.base_encoder(x_canonized)
        reps = reps.reshape(batch_size, -1)
        return self.predictor(reps)


class CNp4CNN(nn.Module):
    """
    Exact CN(p4)-CNN model for Rotated-MNIST.

    Expected input:
        x of shape (B, 1, 28, 28)

    Repo-matching defaults:
        backbone out_channels = 32
        canonization_out_channels = 16
        canonization_num_layers = 3
        canonization_kernel_size = 3
        canonization_beta = 1.0
        num_rotations = 4
        num_classes = 10
    """
    def __init__(
        self,
        device="cuda",
        batch_size=256,
        num_classes=10,
        backbone_out_channels=32,
        canonization_out_channels=16,
        canonization_num_layers=3,
        canonization_kernel_size=3,
        canonization_beta=1.0,
        num_rotations=4,
    ):
        super().__init__()

        self.im_shape = (1, 28, 28)

        base_encoder = CNN(
            in_shape=self.im_shape,
            out_channels=backbone_out_channels,
            num_layers=6,
        ).to(device)

        self.network = EquivariantCanonizationNetwork(
            base_encoder=base_encoder,
            in_shape=self.im_shape,
            num_classes=num_classes,
            canonization_out_channels=canonization_out_channels,
            canonization_num_layers=canonization_num_layers,
            canonization_kernel_size=canonization_kernel_size,
            canonization_beta=canonization_beta,
            num_rotations=num_rotations,
            device=device,
            batch_size=batch_size,
        )

    def forward(self, x):
        return self.network(x)

    def get_canonized_images(self, x):
        return self.network.get_canonized_images(x)

class AverageCNN(nn.Module):
    """
    Returns mean of:
        M(x), M(R90 x), M(R180 x), M(R270 x)

    Efficient implementation:
    run the base model once on a 4x larger batch.
    """
    def __init__(self):
        super().__init__()
        self.base_model = CNN()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected x of shape (B,C,H,W), got {x.shape}")
        if x.shape[-1] != x.shape[-2]:
            raise ValueError("Images must be square")

        B, C, H, W = x.shape

        rots = torch.stack(
            [torch.rot90(x, k=k, dims=(-2, -1)) for k in range(4)],
            dim=1
        )  # (B, 4, C, H, W)

        rots = rots.reshape(4 * B, C, H, W)   # (4B, C, H, W)
        logits = self.base_model(rots)        # (4B, 10)
        logits = logits.reshape(B, 4, -1)     # (B, 4, 10)  
        return logits.mean(dim=1)    