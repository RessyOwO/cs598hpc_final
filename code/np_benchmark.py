import time
import numpy as np

M, N = 8192, 8192
dtype = np.float32
host_a = np.random.rand(M, 1).astype(dtype)
host_b = np.random.rand(1, N).astype(dtype)

t0 = time.perf_counter()
c = host_a + host_b
t1 = time.perf_counter()
bytes_moved = host_a.nbytes + host_b.nbytes + c.nbytes
cpu_bw = bytes_moved / (t1 - t0) / 1e9
print("CPU broadcast:", cpu_bw, "GB/s")
