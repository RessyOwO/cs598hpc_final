# test for upload and download
import numpy as np
from bcast_array import BcastArray

host = np.arange(6, dtype=np.float32).reshape(2, 3)
gpu = BcastArray.from_numpy(host)

round_trip = gpu.get()
print("testing upload and download...")
print("gpu == cpu?", np.allclose(round_trip, host))
print()
