//! Thin safe-ish wrapper over the CUDA-graph FFI shim (`shim.cpp`).
//!
//! Capture protocol (mirrors zerocopy.py `_capture`): move onto a side stream,
//! warm up, then `capture(|| forward_into_static_buffers())`. Replay re-runs the
//! captured kernels against the same static input/output storage as one launch.

use std::ffi::c_void;

extern "C" {
    fn cg_new() -> *mut c_void;
    fn cg_capture_begin(g: *mut c_void);
    fn cg_capture_end(g: *mut c_void);
    fn cg_replay(g: *mut c_void);
    fn cg_free(g: *mut c_void);
    fn cg_set_side_stream();
    fn cg_device_synchronize();
    fn cg_set_cudnn_benchmark(on: bool);
}

/// Enable cuDNN benchmark autotuning (fast conv algos baked in during warmup).
pub fn set_cudnn_benchmark(on: bool) {
    unsafe { cg_set_cudnn_benchmark(on) }
}

// Force the linker to retain the shim object even in code paths that never call
// a graph fn (e.g. `forward-check`). The shim references at::cuda symbols, which
// keeps libtorch_cuda.so as DT_NEEDED; its static initializer registers the CUDA
// backend. Without this, gc-sections / --as-needed drop libtorch_cuda and tch
// reports CUDA as unavailable.
#[used]
static _KEEP_SHIM: unsafe extern "C" fn() = cg_device_synchronize;

/// Switch the current CUDA stream to a fresh non-default stream from the pool
/// (required before graph capture).
pub fn set_side_stream() {
    unsafe { cg_set_side_stream() }
}

pub fn device_synchronize() {
    unsafe { cg_device_synchronize() }
}

pub struct CudaGraph {
    raw: *mut c_void,
}

impl CudaGraph {
    /// Capture `body` (which must run the forward + write the static output
    /// buffers) into a replayable graph. Caller is responsible for being on a
    /// side stream and having warmed up; see `Net::capture_slot`.
    pub fn capture<F: FnOnce()>(body: F) -> CudaGraph {
        let raw = unsafe { cg_new() };
        unsafe { cg_capture_begin(raw) };
        body();
        unsafe { cg_capture_end(raw) };
        CudaGraph { raw }
    }

    pub fn replay(&self) {
        unsafe { cg_replay(self.raw) }
    }
}

impl Drop for CudaGraph {
    fn drop(&mut self) {
        unsafe { cg_free(self.raw) }
    }
}

// The graph handle is only ever used from the single inference thread; the raw
// pointer is not Send/Sync by default, which is what we want (no accidental
// cross-thread replay).
