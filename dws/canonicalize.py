import copy

import numpy as np
import torch


def get_layers(state):
    """
    Return [(weight_key, bias_key), ...] in network order.

    Raw PyTorch Linear weights have shape:
        [out_dim, in_dim]

    A hidden-neuron permutation therefore acts as:
        W_l      -> P_l W_l          (rows)
        b_l      -> P_l b_l
        W_{l+1}  -> W_{l+1} P_l^T    (columns)
    """
    layers = []

    for key, value in state.items():
        if (
            isinstance(value, torch.Tensor)
            and key.endswith(".weight")
            and value.ndim == 2
        ):
            prefix = key[:-len(".weight")]
            bias_key = prefix + ".bias"

            if bias_key not in state:
                raise RuntimeError(f"Missing bias for {key}")

            bias = state[bias_key]

            if bias.ndim != 1:
                raise RuntimeError(
                    f"Expected 1D bias for {bias_key}, got {bias.shape}"
                )

            if value.shape[0] != bias.shape[0]:
                raise RuntimeError(
                    f"Incompatible {key} {value.shape} and "
                    f"{bias_key} {bias.shape}"
                )

            layers.append((key, bias_key))

    if len(layers) < 2:
        raise RuntimeError(
            f"Expected at least 2 linear layers, found {len(layers)}"
        )

    for i in range(len(layers) - 1):
        weight_key, _ = layers[i]
        next_weight_key, _ = layers[i + 1]

        if state[weight_key].shape[0] != state[next_weight_key].shape[1]:
            raise RuntimeError(
                f"Layer mismatch: {weight_key} "
                f"{state[weight_key].shape} -> "
                f"{next_weight_key} "
                f"{state[next_weight_key].shape}"
            )

    return layers


def lexicographic_permutation(key):
    """
    key: [n_neurons, n_features]

    Sort rows lexicographically:
        feature 0, then feature 1, ...
    """
    key_np = key.detach().cpu().numpy()

    # np.lexsort uses the last key as primary.
    perm = np.lexsort(key_np[:, ::-1].T)
    sorted_key = key_np[perm]

    if len(sorted_key) > 1:
        duplicate = np.all(
            sorted_key[1:] == sorted_key[:-1],
            axis=1,
        )

        if duplicate.any():
            ids = np.where(duplicate)[0].tolist()
            raise RuntimeError(
                "Exact canonical-key duplicate(s) at sorted "
                f"positions {ids[:10]}"
            )

    return torch.as_tensor(perm, dtype=torch.long)


def canonicalize_state_dict(state):
    """
    Sequentially canonize hidden-neuron permutations from input to output.

    At hidden layer l, neuron j is sorted by the lexicographic key

        [incoming weights, bias, sorted outgoing weights].

    Sorting the outgoing weights makes that part of the key invariant to the
    still-unknown permutation of hidden layer l+1.

    If the resulting permutation is P_l, we apply it to the rows of W_l and
    b_l, and propagate the same permutation to the columns of W_{l+1}.
    The output layer itself is never sorted.
    """
    state = copy.deepcopy(state)
    layers = get_layers(state)

    for layer_idx in range(len(layers) - 1):
        weight_key, bias_key = layers[layer_idx]
        next_weight_key, _ = layers[layer_idx + 1]

        weight = state[weight_key]
        bias = state[bias_key]
        next_weight = state[next_weight_key]

        incoming = weight

        outgoing_sorted = torch.sort(
            next_weight,
            dim=0,
        ).values.T

        key = torch.cat(
            [
                incoming,
                bias[:, None],
                outgoing_sorted,
            ],
            dim=1,
        )

        perm = lexicographic_permutation(key).to(weight.device)

        state[weight_key] = weight.index_select(0, perm)
        state[bias_key] = bias.index_select(0, perm)
        state[next_weight_key] = next_weight.index_select(1, perm)

    return state


def random_hidden_permutation(state, generator=None):
    """
    Return a functionally equivalent state dict obtained by independently
    permuting every hidden layer.
    """
    state = copy.deepcopy(state)
    layers = get_layers(state)

    for layer_idx in range(len(layers) - 1):
        weight_key, bias_key = layers[layer_idx]
        next_weight_key, _ = layers[layer_idx + 1]

        weight = state[weight_key]
        bias = state[bias_key]
        next_weight = state[next_weight_key]

        n_neurons = weight.shape[0]
        perm = torch.randperm(
            n_neurons,
            generator=generator,
        )

        state[weight_key] = weight.index_select(0, perm)
        state[bias_key] = bias.index_select(0, perm)
        state[next_weight_key] = next_weight.index_select(1, perm)

    return state


def states_equal(a, b):
    if list(a.keys()) != list(b.keys()):
        return False

    for key in a:
        value_a = a[key]
        value_b = b[key]

        if isinstance(value_a, torch.Tensor):
            if not torch.equal(value_a, value_b):
                return False
        elif value_a != value_b:
            return False

    return True
