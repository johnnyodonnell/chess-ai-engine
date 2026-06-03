"""Zero-copy host I/O ring for the rust_pipeline self-play forward (GB10/ATS).

On the GB10 the GPU reads/writes ordinary host memory coherently via ATS (one
LPDDR5X pool). We allocate a small ring of PINNED host buffers (the compiled/
Triton launcher rejects plain *unregistered* host pointers — proven by the
spike — so pinned, not plain) and expose each as a CUDA tensor via a DLPack
capsule tagged kDLCUDA. Rust workers write the encoded batch (bf16) straight
into a slot's input buffer; the net runs on it in place (no H2D); the net's
bf16 outputs are copy_'d into the slot's pinned output buffers (no blocking
cudaMemcpyDtoH); a recorded CUDA event gates the Rust scatter thread's read.

Everything is bf16: the net runs bf16, so f32 outputs carried no extra info —
returning bf16 halves the output bytes and drops the wasted .float() cast.
"""
import ctypes

import torch

# ----------------------------------------------------------------- DLPack glue
class _DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int), ("device_id", ctypes.c_int)]


class _DLDataType(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8),
                ("lanes", ctypes.c_uint16)]


class _DLTensor(ctypes.Structure):
    _fields_ = [("data", ctypes.c_void_p), ("device", _DLDevice),
                ("ndim", ctypes.c_int), ("dtype", _DLDataType),
                ("shape", ctypes.POINTER(ctypes.c_int64)),
                ("strides", ctypes.POINTER(ctypes.c_int64)),
                ("byte_offset", ctypes.c_uint64)]


class _DLManagedTensor(ctypes.Structure):
    pass


_DELETER = ctypes.CFUNCTYPE(None, ctypes.POINTER(_DLManagedTensor))
_DLManagedTensor._fields_ = [("dl_tensor", _DLTensor),
                             ("manager_ctx", ctypes.c_void_p),
                             ("deleter", _DELETER)]

_KDLCUDA = 2
_CODE = {torch.bfloat16: (4, 16), torch.float32: (2, 32)}
_keep = []  # keep ctypes structs alive for the process lifetime (ring is permanent)


def _cuda_view(host_t):
    """Return a CUDA tensor aliasing the pinned host tensor's storage (ATS)."""
    shape = tuple(host_t.shape)
    code, bits = _CODE[host_t.dtype]
    ndim = len(shape)
    shape_arr = (ctypes.c_int64 * ndim)(*shape)
    strides = [1] * ndim
    for i in range(ndim - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    strides_arr = (ctypes.c_int64 * ndim)(*strides)
    mt = _DLManagedTensor()
    mt.dl_tensor.data = ctypes.c_void_p(host_t.data_ptr())
    mt.dl_tensor.device = _DLDevice(_KDLCUDA, 0)
    mt.dl_tensor.ndim = ndim
    mt.dl_tensor.dtype = _DLDataType(code, bits, 1)
    mt.dl_tensor.shape = shape_arr
    mt.dl_tensor.strides = strides_arr
    mt.dl_tensor.byte_offset = 0
    cb = _DELETER(lambda _p: None)
    mt.deleter = cb
    _keep.append((mt, shape_arr, strides_arr, cb, host_t))
    PyCapsule_New = ctypes.pythonapi.PyCapsule_New
    PyCapsule_New.restype = ctypes.py_object
    PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    cap = PyCapsule_New(ctypes.addressof(mt), b"dltensor", None)
    return torch.utils.dlpack.from_dlpack(cap)


# ----------------------------------------------------------------- the ring
class Slot:
    __slots__ = ("input_host", "input_cuda", "logit_host", "logit_cuda",
                 "value_host", "value_cuda", "event")

    def __init__(self, B, in_ch, policy):
        # pinned host buffers (the buffers the GPU reads/writes in place)
        self.input_host = torch.empty(B, in_ch, 8, 8, dtype=torch.bfloat16,
                                      pin_memory=True)
        self.logit_host = torch.empty(B, policy, dtype=torch.bfloat16,
                                      pin_memory=True)
        self.value_host = torch.empty(B, dtype=torch.bfloat16, pin_memory=True)
        # CUDA views aliasing the same storage (what the net runs on / writes to)
        self.input_cuda = _cuda_view(self.input_host)
        self.logit_cuda = _cuda_view(self.logit_host)
        self.value_cuda = _cuda_view(self.value_host)
        self.event = torch.cuda.Event()


class HostRing:
    """N pinned slots. Rust writes inputs + reads outputs by raw pointer; Python
    runs the net into each slot and records its event."""

    def __init__(self, n_slots, B, in_ch=18, policy=4672):
        self.B = B
        self.in_ch = in_ch
        self.policy = policy
        self.n_slots = n_slots
        self.slots = [Slot(B, in_ch, policy) for _ in range(n_slots)]
        # Raw host pointers for the Rust side (one per slot).
        self.input_ptrs = [s.input_host.data_ptr() for s in self.slots]
        self.logit_ptrs = [s.logit_host.data_ptr() for s in self.slots]
        self.value_ptrs = [s.value_host.data_ptr() for s in self.slots]
        # Row strides in ELEMENTS (bf16 u16 units), row-major contiguous.
        self.in_stride = in_ch * 8 * 8     # 1152
        self.logit_stride = policy         # 4672

    def run(self, net, slot_idx, m):
        """Run the net on slot `slot_idx`'s first m rows; write bf16 outputs into
        the slot's pinned output buffers; record the slot's event. Returns the
        slot index back to Rust."""
        s = self.slots[slot_idx]
        logits, values = net(s.input_cuda[:m])
        s.logit_cuda[:m].copy_(logits)     # bf16 -> bf16, GPU writes host buffer
        s.value_cuda[:m].copy_(values)
        s.event.record()
        return slot_idx

    def sync(self, slot_idx):
        """Block until slot `slot_idx`'s forward (incl. output write) completes.
        Called by the Rust scatter thread before it reads the output buffers."""
        self.slots[slot_idx].event.synchronize()
