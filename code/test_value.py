import numpy as np
from bcast_array import BcastArray

OPS = {
    "+": ("__add__", np.add),
    "-": ("__sub__", np.subtract),
    "*": ("__mul__", np.multiply),
    "/": ("__truediv__", np.divide),

    "<": ("__lt__", np.less),
    "<=": ("__le__", np.less_equal),
    "==": ("__eq__", np.equal),
    ">": ("__gt__", np.greater),
    ">=": ("__ge__", np.greater_equal),
    "!=": ("__ne__", np.not_equal),
}

def single_case():
    A = np.arange(35, dtype=np.float32).reshape(5,1,7)
    B = (100 * np.arange(28, dtype=np.float32)).reshape(1,4,7)
    C = A + B

    gpu_C = (BcastArray.from_numpy(A) + BcastArray.from_numpy(B)).get()
    assert np.allclose(gpu_C, C)

def small_ops():
    rng = np.random.default_rng(42)
    # ops = {"+": np.add, "-": np.subtract, "*": np.multiply, "/": np.divide}

    for rank in range(1, 5):
        for _ in range(20):
            shape_a = tuple(rng.integers(1, 5, size=rank))
            shape_b = tuple(rng.integers(1, 5, size=rank))
            try:
                out = np.broadcast_shapes(shape_a, shape_b)
            except ValueError:
                continue

            # avoid divide by 0 error by adding 0.0000001
            A = rng.random(shape_a, dtype=np.float32) + 0.000001 
            B = rng.random(shape_b, dtype=np.float32) + 0.000001

            a_gpu = BcastArray.from_numpy(A)
            b_gpu = BcastArray.from_numpy(B)

            for sym, (o, func) in OPS.items():
                res_gpu = getattr(a_gpu, o)(b_gpu).get()
                assert np.allclose(res_gpu, func(A, B), atol=1e-6), f"op {sym} failed"

def bigger_ops():
    rng = np.random.default_rng(42)
    ops = {"+": np.add, "-": np.subtract, "*": np.multiply, "/": np.divide}

    for rank in range(1, 1000):
        for _ in range(20):
            shape_a = tuple(rng.integers(1, 1000, size=rank))
            shape_b = tuple(rng.integers(1, 1000, size=rank))
            try:
                out = np.broadcast_shapes(shape_a, shape_b)
            except ValueError:
                continue

            # avoid divide by 0 error by adding 0.0000001
            A = rng.random(shape_a, dtype=np.float32) + 0.000001 
            B = rng.random(shape_b, dtype=np.float32) + 0.000001

            a_gpu = BcastArray.from_numpy(A)
            b_gpu = BcastArray.from_numpy(B)

            for sym, (o, func) in OPS.items():
                res_gpu = getattr(a_gpu, o)(b_gpu).get()
                assert np.allclose(res_gpu, func(A, B), atol=1e-6), f"op {sym} failed"

if __name__ == "__main__":
    print("testing 1 simple case...")
    single_case()
    print("passed 1 simple test")
    print()
    print("stress testing on small matrices...")
    small_ops()
    print("passed small matrices binary-op broadcast tests")
    print()
    print("stress testing on bigger matrices...")
    bigger_ops()
    print("passed bigger matrices binary-op broadcast tests")