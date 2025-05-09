# pyopencl does not do broadcasting for you
import numpy as np, pyopencl as cl, pyopencl.array as cl_array
import pyopencl.elementwise as cl_elem

ctx = cl.create_some_context(interactive=False)
queue = cl.CommandQueue(ctx)

# array a has shape (5, 1, 7)
a = np.arange(35, dtype=np.float32).reshape(5, 1, 7)
print("a.shape: ", a.shape)
print(a)
# array b has shape (1, 4, 7)
b = (100 * np.arange(28, dtype=np.float32)).reshape(1, 4, 7)
print("array_b.shape: ", b.shape)
print(b)
print()
print("expected broadcasting shape:", np.broadcast_shapes(a.shape, b.shape))
bcast = a + b
print("broadcasting shape verified. This should be (5, 4, 7): ", bcast.shape)
print()

# send arrays to GPU without shapecheck
gpu_a = cl_array.to_device(queue, a)
gpu_b = cl_array.to_device(queue, b)

try:
    gpu_out = gpu_a + gpu_b
except Exception as e:
    print("PyOpenCL raised an error:", e)