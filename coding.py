"""
Quantization utilities.
"""

import numpy as np
import torch


def uniform_encode(v, s):
    """
    Uniform stochastic quantization encoding. Described in QSGD paper.
    Reference implementation: https://github.com/stsievert/LeanSGD/,
    modified for an arbitrary number of quantization levels.

    Arguments:
        v - tensor
        s - number of quantization levels, int (>= 2)
    """
    if isinstance(v, (torch.Tensor, torch.cuda.FloatTensor)):
        w = v.cpu().numpy()
    elif isinstance(v, np.ndarray):
        w = v
    else:
        raise ValueError("Object passed to encode not ndarray or torch.Tensor")

    norm = np.linalg.norm(v)
    signs = np.sign(w).astype('int')
    probs = np.abs(w) / norm

    # Compute floors
    scaled_probs = probs * s
    floors = np.floor(scaled_probs)  # still floats with numpy
    # Now compute a mask by stochastically deciding 
    # whether to add an add-on to floor (making it ceil):
    remainders = (scaled_probs - floors)
    assert (np.all(remainders < 1) and np.all(remainders > 0))
    mask = np.random.rand(*remainders.shape) < remainders  # bools
    code_bins = floors + mask

    code = {'signs': signs, 'code_bins': code_bins, 'norm': norm}
    return code


def uniform_decode(code, s, device=torch.device("cpu")):
    """
    Arguments:
        code - dictionary
        s - number of quantization levels
    """
    bin_values = code["code_bins"] / s
    v = torch.FloatTensor(code["norm"] * code["signs"] * bin_values)
    v.to(device)
    return v


def uniform_reconstruct(grad_dict, s):
    def on_vector(v, s):
        if isinstance(v, torch.cuda.FloatTensor):
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        code = uniform_encode(v, s=s)
        dec = uniform_decode(code, s=s, device=device)
        return dec
    dec = {k: 0.0 for k in grad_dict.keys()}
    for k in grad_dict:
        dec[k] = on_vector(grad_dict[k], s)
    return dec


def test_uniform_code():
    """Not worth proper unittesting"""
    for shape in ((1, 3, 1), (20,), (10, 20, 30)):
        randomv = torch.rand(shape)
        for s in [5, 10, 30, 1000, 10000000]:
            code = uniform_encode(randomv, s=s)
            dec = uniform_decode(code, s=s)
            err = np.linalg.norm(randomv - dec)
            print("numel={}, s={}, error={:.5f}".format(randomv.numel(), s, err))


if __name__ == "__main__":
    test_uniform_code()

