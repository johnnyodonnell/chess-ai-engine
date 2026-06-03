//! Rust wrapper over the AOTInductor package-loader FFI shim.
//!
//! Loads a `.pt2` (Inductor's fused Triton cubins, no Python) and runs the fused
//! forward. tch tensors bridge as raw `C_tensor` handles (which are `at::Tensor*`
//! under the hood); outputs are copied into caller-owned tch tensors.

use std::ffi::{c_void, CString};

use tch::Tensor;

extern "C" {
    fn aoti_load(path: *const std::os::raw::c_char) -> *mut c_void;
    fn aoti_run(loader: *mut c_void, input: *const c_void, out_logits: *const c_void, out_values: *const c_void);
    fn aoti_free(loader: *mut c_void);
}

pub struct AotiModel {
    raw: *mut c_void,
}

// Used only from the single inference thread.
unsafe impl Send for AotiModel {}

impl AotiModel {
    pub fn load(path: &str) -> AotiModel {
        let c = CString::new(path).unwrap();
        let raw = unsafe { aoti_load(c.as_ptr()) };
        assert!(!raw.is_null(), "aoti_load returned null for {path}");
        AotiModel { raw }
    }

    /// Run the fused forward on `input` [B,18,8,8] bf16 CUDA; write results into
    /// `out_logits` [B,4672] and `out_values` [B] (caller-owned, bf16 CUDA).
    pub fn run(&self, input: &Tensor, out_logits: &Tensor, out_values: &Tensor) {
        unsafe {
            aoti_run(
                self.raw,
                input.as_ptr() as *const c_void,
                out_logits.as_ptr() as *const c_void,
                out_values.as_ptr() as *const c_void,
            );
        }
    }
}

impl Drop for AotiModel {
    fn drop(&mut self) {
        unsafe { aoti_free(self.raw) }
    }
}
