import numpy as np
from bcast_array import get_bshape, get_bstride

print("checking toy shape...")
# toy test
shape_a = (5, 1, 7)
shape_b = (1, 4, 7)

out_shape, same_a, same_b = get_bshape(shape_a, shape_b)
assert out_shape == (5, 4, 7)
# axis 1 broadcast
assert same_a == (True,  False, True)
# axis 0 broadcast
assert same_b == (False, True,  True)
print("passed toy shape check")
print()

print("checking toy strides...")
# check strides
item = 4
strides_a = (28, 28, 4)
strides_b = (112, 28, 4)

ba = get_bstride(shape_a, strides_a, out_shape)
bb = get_bstride(shape_b, strides_b, out_shape)
assert ba == (28, 0, 4)
assert bb == (0, 28, 4)
print("passed toy stride check")
print()

print("stress testing with 100 cases of small size matrices...")
# stress testing
rng = np.random.default_rng(0)
for _ in range(100):
    rank = rng.integers(1, 6)
    shape_a = tuple(rng.integers(1, 6, size=rank))
    shape_b = tuple(rng.integers(1, 6, size=rng.integers(1, 6)))

    try:
        ref_shape = np.broadcast_shapes(shape_a, shape_b)
    except ValueError:
        try:
            get_bshape(shape_a, shape_b)
        except ValueError:
            continue
        else:
            raise AssertionError("BIBI")
    else:
        out_shape, _, _ = get_bshape(shape_a, shape_b)
        assert out_shape == ref_shape
        item = 4
        strides_a = tuple(np.lib.stride_tricks.as_strided(np.empty(shape_a, np.uint8), shape_a, writeable=False).strides)
        strides_b = tuple(np.lib.stride_tricks.as_strided(np.empty(shape_b, np.uint8), shape_b, writeable=False).strides)

        new_a = get_bstride(shape_a, strides_a, ref_shape)
        new_b = get_bstride(shape_b, strides_b, ref_shape)
        pad_a = len(ref_shape) - len(shape_a)
        pad_b = len(ref_shape) - len(shape_b)

    for osize, outsize, nstr in zip((1,) * pad_a + shape_a, ref_shape, new_a):
        if osize == 1 and outsize > 1:
            assert nstr == 0

    for osize, outsize, nstr in zip((1,) * pad_b + shape_b, ref_shape, new_b):
        if osize == 1 and outsize > 1:
            assert nstr == 0

print("passed stress testing")
print()

