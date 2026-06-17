"""
Complexity.py
=============
Parameter count, FLOP estimate, and the asymptotic time-complexity expression
for the computational-complexity table. The 'ours' row is computed directly
from the built model and the constants in Config.py.
"""
import numpy as np
import Config


def count_parameters(model):
    return int(sum(int(w.shape.num_elements()) for w in model.trainable_weights))


def estimate_flops(model):
    total = 0
    for layer in model.layers:
        name = layer.__class__.__name__
        try:
            out = layer.output_shape
        except Exception:
            continue
        if name == "Conv2D":
            k = int(np.prod(layer.kernel_size)); cin = layer.input_shape[-1] or 1
            total += 2 * out[1] * out[2] * out[3] * cin * k
        elif name == "Dense":
            total += 2 * (layer.input_shape[-1] or 1) * (out[-1] or 1)
        elif name == "LSTM":
            u = layer.units; cin = layer.input_shape[-1] or 1
            steps = layer.input_shape[1] or 1
            total += 2 * 4 * u * (cin + u) * steps
    return int(total)


def big_o():
    c = Config.COMPLEXITY
    value = c["N_t"] * c["S_i"] * (c["T_e"] * c["F_e"] + c["T_f"] * c["F_f"]) * c["H"] ** 2
    return "O(N_t * S_i * (T_e*F_e + T_f*F_f) * H^2)", int(value)


def summary(model=None):
    expr, value = big_o()
    print(f"Time complexity : {expr}")
    print(f"               ~ {value:.3e} operations")
    if model is not None:
        print(f"Parameters      : {count_parameters(model):,}")
        try:
            print(f"FLOPs (approx.) : {estimate_flops(model):,}")
        except Exception as e:
            print(f"FLOPs (approx.) : n/a ({e})")
