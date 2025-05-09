# show wrapper goal:
# figure out the common broadcast shape
# create stride-0 view
# hand those views to a Loopy kernel
# this step i want to show stride 0 view is possible on GPU
import numpy as np, pyopencl as cl, pyopencl.array as cl_array
import loopy as lp

# show broadcasting is essentially achieved by using 0 stride to duplicate values
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

# now move it onto the GPU
ctx = cl.create_some_context(interactive=False)
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags

# copy the contiguous array onto GPU
a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)

# handmake the array of shape 5, 1, 7 with buf data and stride 0 on second axis
gpu_a = cl_array.Array(queue, shape=(5, 1, 7), dtype=a.dtype, data=a_buf, strides=(28, 0, 4))

print("gpu_a shape:", gpu_a.shape)
print("gpu_a strides:", gpu_a.strides)

b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
gpu_b = cl_array.Array(queue, shape=(1, 4, 7), dtype=b.dtype, data=b_buf, strides=(0, 28, 4))

print("gpu_b shape:", gpu_b.shape)
print("gpu_b strides:", gpu_b.strides)

# loopy kernel
knl = lp.make_kernel(
    "{[i,j,k]: 0<=i<n0 and 0<=j<n1 and 0<=k<n2}",
    "out[i,j,k] = a[i,0,k] + b[0,j,k]",
    [
        lp.GlobalArg("a", dtype=np.float32,
                     shape=("n0", 1, "n2"), strides=lp.auto),
        lp.GlobalArg("b", dtype=np.float32,
                     shape=(1, "n1", "n2"), strides=lp.auto),
        lp.GlobalArg("out", dtype=np.float32,
                     shape=("n0", "n1", "n2"), strides=lp.auto),
        lp.ValueArg("n0", np.int32),
        lp.ValueArg("n1", np.int32),
        lp.ValueArg("n2", np.int32),
    ],
    name="broadcast_add_3d",
    # get rid of warning
    lang_version=(2018, 2)
)

gpu_out = cl_array.empty(queue, (5, 4, 7), np.float32)

# run
knl(queue, a=gpu_a, b=gpu_b, out=gpu_out, n0=5, n1=4, n2=7)
queue.finish()

# verify results
cpu_ref = bcast
print()
print("gpu == numpy? ",
      np.allclose(gpu_out.get(), cpu_ref))
print("gpu:\n", gpu_out.get())