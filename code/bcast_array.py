import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
import loopy as lp
from numbers import Number
import re


global_queue = None
def get_queue():
    global global_queue
    if global_queue is None:
        ctx = cl.create_some_context()
        global_queue = cl.CommandQueue(ctx)
    return global_queue

# monkey patch at import to override numpy's 32-dim limit
def broadcast_shapes_unlimited(*shapes):
    def pairwise(a, b):
        la, lb = len(a), len(b)
        out = []
        for i in range(max(la, lb)):
            da = a[-1-i] if i < la else 1
            db = b[-1-i] if i < lb else 1
            if da == db or da == 1 or db == 1:
                out.append(max(da, db))
            else:
                raise ValueError(f"shapes {a} and {b} are not broadcastable")
        return tuple(reversed(out))

    res = shapes[0]
    for s in shapes[1:]:
        res = pairwise(res, s)
    return res

np.broadcast_shapes = broadcast_shapes_unlimited

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
    orig_shape = (1,) * pad + orig_shape
    orig_strides = (0,) * pad + orig_strides

    new = [0 if (s==1 and o>1) else st for s, st, o in zip(orig_shape, orig_strides, out_shape)]
    return tuple(new)


# map operator +, -, *, / to string name
def op2name(op: str) -> str:
    return {
        "+": "add",
        "-": "sub",
        "*": "mul",
        "/": "div",
        "<":  "lt",
        "<=": "le",
        "==": "eq",
        ">":  "gt",
        ">=": "ge",
        "!=": "ne",
    }.get(op, re.sub(r"\W|^(?=\d)", "_", op))

_kernel_cache: dict[tuple[str, int, np.dtype], callable] = {}
# make loopy kernel
def make_binop_kernel(op: str, rank: int, dtype: np.dtype):
    key = (op, rank, dtype)
    if key in _kernel_cache:
        return _kernel_cache[key]

    # build index names and domain
    dims = [f"i{k}"  for k in range(rank)]
    dimvars = [f"n{k}"  for k in range(rank)]
    dom_idxs = ",".join(dims)
    dom_conds= " and ".join(f"0<={d}<{v}" for d, v in zip(dims, dimvars))
    domain = f"{{[{dom_idxs}]: {dom_conds}}}"

    # compute out
    out = f"out[{dom_idxs}] = a[{dom_idxs}] {op} b[{dom_idxs}]"

    # arguments
    # for each array we accept data pointer + strides for the axis
    a_strides = tuple(f"a_stride_{rank-1-k}" for k in range(rank))
    b_strides = tuple(f"b_stride_{rank-1-k}" for k in range(rank))
    out_strides = tuple(f"out_stride_{rank-1-k}" for k in range(rank))

    args = [
        lp.GlobalArg("a", dtype=dtype, shape=tuple(dimvars), strides=a_strides),
        lp.GlobalArg("b", dtype=dtype, shape=tuple(dimvars), strides=b_strides),
        lp.GlobalArg("out", dtype=dtype, shape=tuple(dimvars), strides=out_strides),
    ]

    # valueargs for each stride and dim
    for arr in ("a", "b", "out"):
        for i in range(rank):
            args.append(lp.ValueArg(f"{arr}_stride_{i}", np.int32))
    for i in range(rank):
        args.append(lp.ValueArg(f"n{i}", np.int32))

    # build and cache kernel
    name = f"bcast_{op2name(op)}_{rank}d"
    knl = lp.make_kernel(domain, out, args, name=name, lang_version=(2018, 2))

    _kernel_cache[key] = knl
    return knl

# convert byte strides (numpy/cl) to element strides (loopy)
def as_elem_strides(byte_strides, itemsize):
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
        return cl_array.Array(self.queue, out_shape, self.dtype, data=self.data, strides=new_strides)

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
        knl = make_binop_kernel(op, rank, self.dtype)

        launch = {}
        for name, view in (("a", a_view), ("b", b_view), ("out", out)):
            launch[name] = view
            elt_strides = as_elem_strides(view.strides, self.dtype.itemsize)
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
    def __lt__(self, other): return self.binop(other, "<")
    def __le__(self, other): return self.binop(other, "<=")
    def __eq__(self, other): return self.binop(other, "==")
    def __gt__(self, other): return self.binop(other, ">")
    def __ge__(self, other): return self.binop(other, ">=")
    def __ne__(self, other): return self.binop(other, "!=")

