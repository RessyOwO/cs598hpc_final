import os
import numpy as np
import pyopencl as cl
import pyopencl.array as cla

from bcast_array import BcastArray, make_binop_kernel, as_elem_strides

os.environ.setdefault("PYOPENCL_CTX", "1:0")
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx,
        properties=cl.command_queue_properties.PROFILING_ENABLE)
print("device:", queue.device.name)

M, N = 8192, 8192
dtype = np.float32
a = np.random.rand(M, 1).astype(dtype)
b = np.random.rand(1, N).astype(dtype)

a_bcast = BcastArray.from_numpy(a, queue)
b_bcast = BcastArray.from_numpy(b, queue)

out_shape = (M, N)
a_view = a_bcast.broadcast_view(out_shape)
b_view = b_bcast.broadcast_view(out_shape)
# output
c = cla.empty(queue, out_shape, dtype)

rank = 2
# translation unit
tu = make_binop_kernel("+", rank, dtype)
# JIT + cache
exec_knl = tu.executor(queue) 

bytes_per = dtype().itemsize
launch = {"a": a_view, "b": b_view, "out": c, "n0": np.int32(M), "n1": np.int32(N)}
for name, arr in (("a", a_view), ("b", b_view), ("out", c)):
    s1, s0 = as_elem_strides(arr.strides, bytes_per)
    launch[f"{name}_stride_1"] = np.int32(s1)
    launch[f"{name}_stride_0"] = np.int32(s0)

# warmup
evt = exec_knl(queue, **launch)[0]
evt.wait()

# time
best = 1e9
for _ in range(20):
    evt = exec_knl(queue, **launch)[0]
    evt.wait()
    t = (evt.profile.end - evt.profile.start) * 1e-9
    best = min(best, t)

# bandwidth
bytes_moved = a.nbytes + b.nbytes + M * N * bytes_per
gbps = bytes_moved / best / 1e9
peak = 652.8  # peak GB/s from google gemini

print(f"\nbest time   : {best*1e3:7.2f} ms")
print(f"bytes moved : {bytes_moved/1e9:7.2f} GB")
print(f"BW achieved : {gbps:7.1f} GB/s   ({gbps/peak*100:4.1f}% peak)")
