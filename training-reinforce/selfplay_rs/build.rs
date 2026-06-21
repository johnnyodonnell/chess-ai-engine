// Force-load the PyTorch CUDA backend so `tch::Cuda::is_available()` is true.
//
// tch (LIBTORCH_USE_PYTORCH=1) links libtorch but, under the linker's default
// --as-needed, drops torch_cuda/c10_cuda when no CUDA symbol is referenced
// *directly* from Rust (tch routes CUDA ops through the dynamic dispatcher). If
// those libs are dropped their static initializers never run and CUDA looks
// unavailable. So we re-add them with --no-as-needed before the -l flags. This
// mirrors the link section of training/selfplay_rs/build.rs (which already
// builds + runs on asus), minus its C++ CUDA-graph shim (not used here).
use std::process::Command;

fn py() -> String {
    std::env::var("SELFPLAY_PYTHON").unwrap_or_else(|_| "python3".to_string())
}

fn py_out(code: &str) -> String {
    let o = Command::new(py())
        .args(["-c", code])
        .output()
        .expect("failed to run python for build config (set SELFPLAY_PYTHON)");
    if !o.status.success() {
        panic!("python build query failed: {}", String::from_utf8_lossy(&o.stderr));
    }
    String::from_utf8(o.stdout).unwrap().trim().to_string()
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    let torch_lib =
        py_out("import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))");
    println!("cargo:rustc-link-search=native={}", torch_lib.trim());

    // libcudart ships in the pip nvidia-cuda-runtime wheel as libcudart.so.12
    // (no unversioned symlink), so link it by exact name from that wheel's lib.
    let cudart_dir = py_out(
        "import os, nvidia.cuda_runtime as r; print(os.path.join(os.path.dirname(r.__file__), 'lib'))",
    );
    println!("cargo:rustc-link-search=native={}", cudart_dir.trim());

    println!("cargo:rustc-link-arg=-Wl,--no-as-needed");
    println!("cargo:rustc-link-arg=-lc10_cuda");
    println!("cargo:rustc-link-arg=-ltorch_cuda");
    println!("cargo:rustc-link-arg=-l:libcudart.so.12");
}
