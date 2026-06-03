// Minimal C FFI over libtorch's CUDA-graph + stream API, which tch-rs does not
// expose. Symbols resolve against the torch_cuda / c10_cuda libs that tch links.
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAFunctions.h>

extern "C" {

void* spike_graph_new() {
    return reinterpret_cast<void*>(new at::cuda::CUDAGraph());
}

void spike_graph_capture_begin(void* g) {
    reinterpret_cast<at::cuda::CUDAGraph*>(g)->capture_begin();
}

void spike_graph_capture_end(void* g) {
    reinterpret_cast<at::cuda::CUDAGraph*>(g)->capture_end();
}

void spike_graph_replay(void* g) {
    reinterpret_cast<at::cuda::CUDAGraph*>(g)->replay();
}

void spike_graph_free(void* g) {
    delete reinterpret_cast<at::cuda::CUDAGraph*>(g);
}

// Switch the current CUDA stream to a fresh non-default stream from the pool.
// CUDAGraph::capture_begin requires a non-default capture stream.
void spike_set_side_stream() {
    auto s = c10::cuda::getStreamFromPool(false);
    c10::cuda::setCurrentCUDAStream(s);
}

void spike_device_synchronize() {
    c10::cuda::device_synchronize();
}

}  // extern "C"
