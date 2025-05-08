# numpy broadcasting rules
import numpy as np

# 2 arrays with differnt but compatible shapes
# a has shape (3, 1, 4)
a = np.ones((3, 1, 4))
# b has shape (4,)
b = np.arange(4)
# when we do a + b, b is treated as (1, 1, 4)
c = a + b
# check shape
print("shape checking...")
print("a.shape:", a.shape)
print("b.shape:", b.shape, "(interpreted as (1,1,4))")
print("c.shape:", c.shape)
print()
# check value
print("value checking...")
print("a value:")
print(a)
print("b value:")
print(b)
print("c value:")
print(c)
print()

# numpy tells us what the shape should be after broadcasting
# https://numpy.org/doc/2.2/reference/generated/numpy.broadcast_shapes.html
out_shape = np.broadcast_shapes(a.shape, b.shape)
print("np.broadcast_shapes result:", out_shape)
print()

# looking at 10 more examples
np.random.seed(3)
for i in range(10):
    # random 1-4 dimensions
    dim1, dim2 = np.random.randint(1, 5, size=2)
    # each num between 1-5 in each dimension
    shape1 = np.random.randint(1, 6, size=dim1)
    shape2 = np.random.randint(1, 6, size=dim2)

    try:
        out = np.broadcast_shapes(shape1, shape2)
        status = f"OK -> {out}"
    except ValueError as e:
        status = "incompatible"

    print(f"{i+1}. {tuple(map(int, shape1))} vs {tuple(map(int, shape2))}")
    print(f"   {status}")
