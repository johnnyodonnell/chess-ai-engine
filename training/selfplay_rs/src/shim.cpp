// Minimal C FFI over libtorch's CUDA-graph + stream API (not exposed by tch-rs).
// Symbols resolve against the torch_cuda / c10_cuda libs that tch links.
#include <ATen/cuda/CUDAGraph.h>
#include <ATen/Context.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAFunctions.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <torch/csrc/inductor/aoti_runner/model_container_runner.h>
#include <cuda_runtime.h>
#include <string>
#include <unordered_map>
#include <vector>

extern "C" {

// --- CUDA events: gate the scatter thread's readback on the forward's
// completion so it overlaps the inference thread's next forward. Recorded on the
// thread's current CUDA stream.
void* cg_event_new() {
    cudaEvent_t e;
    cudaEventCreateWithFlags(&e, cudaEventDisableTiming);
    return reinterpret_cast<void*>(e);
}
void cg_event_record(void* e) {
    cudaEventRecord(reinterpret_cast<cudaEvent_t>(e),
                    c10::cuda::getCurrentCUDAStream().stream());
}
void cg_event_sync(void* e) {
    cudaEventSynchronize(reinterpret_cast<cudaEvent_t>(e));
}
void cg_event_free(void* e) {
    cudaEventDestroy(reinterpret_cast<cudaEvent_t>(e));
}

// --- AOTInductor package loader (the fused, Python-free forward) -------------
// tch tensor handles (C_tensor*) ARE at::Tensor* under the hood, so we
// reinterpret them directly. Outputs are copy_'d into Rust-owned tensors so all
// lifetime stays on the tch side.
void* aoti_load(const char* path) {
    // run_single_threaded=true runs the model inline on the calling thread instead
    // of via AOTI's thread pool. The pooled path issues an op that's illegal during
    // CUDA-graph capture ("operation not permitted when stream is capturing", pytorch
    // #158834); the single-threaded path is capture-safe, letting us graph the AOTI
    // forward (and it has no downside for our single-stream inference).
    return reinterpret_cast<void*>(new torch::inductor::AOTIModelPackageLoader(
        std::string(path), "model", /*run_single_threaded=*/true));
}

void aoti_run(void* loader_, const void* in_, void* out_logits_, void* out_values_) {
    auto* loader = reinterpret_cast<torch::inductor::AOTIModelPackageLoader*>(loader_);
    const at::Tensor& in = *reinterpret_cast<const at::Tensor*>(in_);
    std::vector<at::Tensor> outs = loader->run({in});
    reinterpret_cast<at::Tensor*>(const_cast<void*>(out_logits_))->copy_(outs[0]);
    reinterpret_cast<at::Tensor*>(const_cast<void*>(out_values_))->copy_(outs[1]);
}

void aoti_free(void* loader_) {
    delete reinterpret_cast<torch::inductor::AOTIModelPackageLoader*>(loader_);
}

// In-place weight refresh (no recompile). names[] are the MANGLED constant names
// (dotted FQN with '.'->'_'); tensors[] are tch C_tensor handles (at::Tensor*) on
// CUDA, bf16. Push into the INACTIVE constant buffer, re-derive the _FOLDED_CONST_*
// tensors from the new primaries (requires the .pt2 compiled with
// use_runtime_constant_folding=True), then atomically swap it live. Runs on the
// caller's current CUDA stream, so it serializes safely against the inference
// thread's forwards (single reader; the active buffer is untouched until swap).
void aoti_swap_weights(void* loader_, const char** names, const void** tensors, int n) {
    auto* loader = reinterpret_cast<torch::inductor::AOTIModelPackageLoader*>(loader_);
    auto* runner = loader->get_runner();
    std::unordered_map<std::string, at::Tensor> map;
    map.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; i++) {
        map[std::string(names[i])] = *reinterpret_cast<const at::Tensor*>(tensors[i]);
    }
    runner->update_constant_buffer(map, /*use_inactive=*/true, /*validate_full_updates=*/false);
    auto stream = reinterpret_cast<AOTInductorStreamHandle>(
        c10::cuda::getCurrentCUDAStream().stream());
    runner->run_const_fold(/*use_inactive=*/true, stream);
    runner->swap_constant_buffer();
}

// Enable cuDNN benchmark (algorithm autotuning) — picks fast conv algorithms on
// the first calls (during warmup), which then get baked into the captured graph.
// tch does not expose this; the deterministic default algos are much slower at
// B=512.
void cg_set_cudnn_benchmark(bool on) {
    at::globalContext().setBenchmarkCuDNN(on);
}

void* cg_new() {
    return reinterpret_cast<void*>(new at::cuda::CUDAGraph());
}

void cg_capture_begin(void* g) {
    reinterpret_cast<at::cuda::CUDAGraph*>(g)->capture_begin();
}

void cg_capture_end(void* g) {
    reinterpret_cast<at::cuda::CUDAGraph*>(g)->capture_end();
}

void cg_replay(void* g) {
    reinterpret_cast<at::cuda::CUDAGraph*>(g)->replay();
}

void cg_free(void* g) {
    delete reinterpret_cast<at::cuda::CUDAGraph*>(g);
}

// Switch the current CUDA stream to a fresh non-default stream from the pool.
// CUDAGraph::capture_begin requires a non-default capture stream.
void cg_set_side_stream() {
    auto s = c10::cuda::getStreamFromPool(false);
    c10::cuda::setCurrentCUDAStream(s);
}

void cg_device_synchronize() {
    c10::cuda::device_synchronize();
}

}  // extern "C"
