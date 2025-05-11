import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
import loopy as lp
from numbers import Number
from collections import OrderedDict
import re


global_queue = None
def get_queue():
    global global_queue
    if global_queue is None:
        ctx = cl.create_some_context()
        global_queue = cl.CommandQueue(ctx)
    return global_queue

# helpers
# figure out what is the bshape
def get_bshape(shape_a, shape_b):
    out_shape = np.broadcast_shapes(shape_a, shape_b)

    pad_a = len(out_shape) - len(shape_a)
    pad_b = len(out_shape) - len(shape_b)
    shape_a = (1,) * pad_a + shape_a
    shape_b = (1,) * pad_b + shape_b

    # sanity check to see which axis is in need of broadcasting
    same_a = tuple(a == o for a, o in zip(shape_a, out_shape))
    same_b = tuple(b == o for b, o in zip(shape_b, out_shape))

    return out_shape, same_a, same_b

# figure out what is the bstride
def get_bstride(orig_shape, orig_strides, out_shape):
    pad = len(out_shape) - len(orig_shape)
    orig_shape   = (1,) * pad + orig_shape
    orig_strides = (0,) * pad + orig_strides

    new = [0 if (s==1 and o>1) else st for s, st, o in zip(orig_shape, orig_strides, out_shape)]
    return tuple(new)


_kernel_cache: dict[tuple[str, int, np.dtype], callable] = {}

# map operator +, -, *, / to string name
def _op2name(op: str) -> str:
    return {
        "+": "add",
        "-": "sub",
        "*": "mul",
        "/": "div",
    }.get(op, re.sub(r"\W|^(?=\d)", "_", op))

# make loopy kernel
def _get_kernel(op: str, rank: int, dtype: np.dtype):
    # TODO: make loopy kernel
    return

# convert byte strides (numpy/cl) to element strides (loopy)
def _as_elem_strides(byte_strides: tuple[int], itemsize: int) -> tuple[int]:
    """Convert NumPy/CL byte-strides ➜ element-strides expected by Loopy."""
    return tuple(0 if s == 0 else s // itemsize for s in byte_strides)




class BcastArray:
    def __init__(self, buf, shape, strides, dtype, queue=None):
        self.data = buf
        self.shape = tuple(shape)
        self.strides = tuple(strides)
        self.dtype = np.dtype(dtype)
        self.queue = queue if queue else get_queue()

    @classmethod
    # upload to GPU
    def from_numpy(cls, arr, queue=None):
        q = queue or get_queue()
        mf = cl.mem_flags
        buf = cl.Buffer(q.context, mf.COPY_HOST_PTR, hostbuf=arr)
        return cls(buf, arr.shape, arr.strides, arr.dtype, q)

    # download to CPU
    def get(self):
        view = cl_array.Array(self.queue, self.shape, self.dtype, data=self.data, strides=self.strides)
        return view.get()

    # handmake the array of broadcast shape with buf data and right strides
    def broadcast_view(self, out_shape):
        new_strides = get_bstride(self.shape, self.strides, out_shape)
        return cl_array.Array(self.queue, out_shape, self.dtype,
                              data=self.data, strides=new_strides)

    # binop function
    def binop(self, other, op: str):
        # type check - only work if bcastarray binop with bcastarray or number
        if isinstance(other, Number):
            other = BcastArray.from_numpy(np.array(other, self.dtype), self.queue)
        if not isinstance(other, BcastArray):
            raise TypeError("Operand must be BcastArray or scalar")

        out_shape, _, _ = get_bshape(self.shape, other.shape)
        rank = len(out_shape)

        # build broadcast views
        a_view = self.broadcast_view(out_shape)
        b_view = other.broadcast_view(out_shape)
        out = cl_array.empty(self.queue, out_shape, self.dtype)

        # get kernel and construct kwargs
        knl = _get_kernel(op, rank, self.dtype)

        launch = {}
        for name, view in (("a", a_view), ("b", b_view), ("out", out)):
            launch[name] = view.data
            elt_strides = _as_elem_strides(view.strides, self.dtype.itemsize)
            for ax, st in enumerate(elt_strides):
                launch[f"{name}_stride_{rank-1-ax}"] = np.int32(st)
        for k, n in enumerate(out_shape):
            launch[f"n{k}"] = np.int32(n)

        # call kernel with arguments
        knl(self.queue, **launch)
        return BcastArray(out.data, out.shape, out.strides, self.dtype, self.queue)

    # operations
    def __add__(self, other): return self.binop(other, "+")
    def __sub__(self, other): return self.binop(other, "-")
    def __mul__(self, other): return self.binop(other, "*")
    def __truediv__(self, other): return self.binop(other, "/")