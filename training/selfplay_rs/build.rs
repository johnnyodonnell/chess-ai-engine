// Compile the CUDA-graph FFI shim against the box's PyTorch libtorch headers.
// tch (with LIBTORCH_USE_PYTORCH=1) links the torch libs into the final binary,
// so the shim's at::cuda::CUDAGraph symbols resolve at link time. See the
// Stage-1 spike notes in [[rust-selfplay-port]] for the gotchas reproduced here.
use std::process::Command;

fn py() -> String {
    std::env::var("SELFPLAY_PYTHON").unwrap_or_else(|_| "python3".to_string())
}

fn py_out(code: &str) -> String {
    let o = Command::new(py())
        .args(["-c", code])
        .output()
        .expect("failed to run python for build config");
    if !o.status.success() {
        panic!("python build query failed: {}", String::from_utf8_lossy(&o.stderr));
    }
    String::from_utf8(o.stdout).unwrap().trim().to_string()
}

fn main() {
    println!("cargo:rerun-if-changed=src/shim.cpp");
    println!("cargo:rerun-if-changed=build.rs");

    let includes =
        py_out("import torch.utils.cpp_extension as c; print('\\n'.join(c.include_paths()))");
    // torch headers #include "crt/host_config.h", absent from the pip
    // nvidia-cuda-runtime wheel — pick a CUDA include dir that has both
    // cuda_runtime.h and crt/ (the triton-bundled cu12 set qualifies).
    let cuda_inc = py_out(
        "import os, sysconfig, glob\n\
         sp = sysconfig.get_paths()['purelib']\n\
         hits = glob.glob(os.path.join(sp, '**/include/crt/host_config.h'), recursive=True)\n\
         print(os.path.dirname(os.path.dirname(hits[0])) if hits else '/usr/local/cuda/include')",
    );
    let cxx11 = py_out("import torch; print(1 if torch.compiled_with_cxx11_abi() else 0)");

    let mut b = cc::Build::new();
    b.cpp(true).std("c++17").file("src/shim.cpp");
    for line in includes.lines() {
        let p = line.trim();
        if !p.is_empty() {
            b.include(p);
        }
    }
    b.include(cuda_inc.trim());
    b.flag(&format!("-D_GLIBCXX_USE_CXX11_ABI={}", cxx11.trim()));
    b.flag("-DTORCH_API_INCLUDE_EXTENSION_H");
    b.flag_if_supported("-Wno-unused-parameter");
    b.compile("cudagraph_shim");

    // The shim references c10_cuda / torch_cuda symbols that tch does not link.
    let torch_lib =
        py_out("import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))");
    // It ships as libcudart.so.12 (no unversioned symlink), so link by exact name.
    let cudart_dir = py_out(
        "import os, nvidia.cuda_runtime as r; print(os.path.join(os.path.dirname(r.__file__), 'lib'))",
    );
    println!("cargo:rustc-link-search=native={}", torch_lib.trim());
    println!("cargo:rustc-link-search=native={}", cudart_dir.trim());
    // Force-load the CUDA backend libs with --no-as-needed *before* the -l flags,
    // so they're kept even in a binary that references no CUDA symbol directly
    // (evaluate_rs). Otherwise the linker's default --as-needed drops torch_cuda,
    // its static initializers never run, and tch::Cuda::is_available() == false
    // (the self-play bin only kept them incidentally via the CUDA-graph shim).
    println!("cargo:rustc-link-arg=-Wl,--no-as-needed");
    println!("cargo:rustc-link-arg=-lc10_cuda");
    println!("cargo:rustc-link-arg=-ltorch_cuda");
    println!("cargo:rustc-link-arg=-l:libcudart.so.12");
}
