#include "pointnet_inference_engine.h"
#include <onnxruntime_cxx_api.h>
#include <cuda_runtime_api.h>
#include <chrono>
#include <stdexcept>
#include <cstring>
#include <iostream>
#include <mutex>
#include <condition_variable>

#define CUDA_CHECK(expr) \
    do { \
        cudaError_t _err = (expr); \
        if (_err != cudaSuccess) { \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ \
                      << ": " << cudaGetErrorString(_err) << std::endl; \
        } \
    } while(0)

namespace swine3d {

struct CudaMemoryPool {
    float* d_input = nullptr;
    float* d_output = nullptr;
    float* h_pinned_input = nullptr;
    float* h_pinned_output = nullptr;
    size_t input_bytes = 0;
    size_t output_bytes = 0;
    int max_batch = 0;
    int num_points = 0;
    int num_keypoints = 0;
    int coord_dim = 4;
    bool allocated = false;

    size_t inputStrideBytes() const {
        return static_cast<size_t>(num_points) * 3 * sizeof(float);
    }
    size_t outputStrideBytes() const {
        return static_cast<size_t>(num_keypoints) * coord_dim * sizeof(float);
    }
};

struct PointNetInferenceEngine::Impl {
    Ort::Env env{nullptr};
    Ort::SessionOptions session_options{nullptr};
    std::unique_ptr<Ort::Session> session;
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::RunOptions run_options{nullptr};

    std::vector<std::string> input_names_str;
    std::vector<std::string> output_names_str;
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;

    std::vector<int64_t> input_shape;
    std::vector<int64_t> output_shape;

    int expected_point_count = DEFAULT_POINT_COUNT;
    int keypoint_count = DEFAULT_KEYPOINT_COUNT;
    int max_batch_size = MAX_BATCH_SIZE;
    int device_id = 0;
    int coord_dim = 4;

    bool model_loaded = false;
    bool use_trt = false;
    bool cuda_pool_allocated = false;

    float last_inference_ms = 0.0f;
    float last_h2d_ms = 0.0f;
    float last_d2h_ms = 0.0f;

    bool profiling = false;
    int intra_threads = 4;
    float oom_threshold_mb = DEFAULT_OOM_THRESHOLD_MB;

    CudaMemoryPool cuda_pool;

    cudaStream_t cuda_stream = nullptr;
    std::mutex inference_mutex;

    std::atomic<bool> shutdown_requested{false};
    Ort::MemoryInfo cuda_memory_info{nullptr};
    Ort::MemoryInfo cpu_pinned_memory_info{nullptr};

    std::unique_ptr<Ort::IoBinding> io_binding;

    std::vector<int64_t> current_input_shape;
    std::vector<int64_t> current_output_shape;
};

PointNetInferenceEngine::PointNetInferenceEngine()
    : impl_(std::make_unique<Impl>()) {}

PointNetInferenceEngine::~PointNetInferenceEngine() {
    shutdown();
}

void PointNetInferenceEngine::shutdown() {
    std::lock_guard<std::mutex> lock(impl_->inference_mutex);
    if (impl_->shutdown_requested.exchange(true)) return;

    synchronizeCuda();

    freeStaticMemoryPool();

    impl_->io_binding.reset();
    impl_->session.reset();

    if (impl_->cuda_stream != nullptr) {
        CUDA_CHECK(cudaStreamDestroy(impl_->cuda_stream));
        impl_->cuda_stream = nullptr;
    }

    impl_->model_loaded = false;
    impl_->cuda_pool_allocated = false;
}

void PointNetInferenceEngine::synchronizeCuda() {
    if (impl_->cuda_stream != nullptr) {
        CUDA_CHECK(cudaStreamSynchronize(impl_->cuda_stream));
    } else {
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

bool PointNetInferenceEngine::allocateStaticMemoryPool(
    int max_batch, int num_points, int num_keypoints) {

    if (impl_->cuda_pool_allocated) {
        freeStaticMemoryPool();
    }

    impl_->cuda_pool.max_batch = max_batch;
    impl_->cuda_pool.num_points = num_points;
    impl_->cuda_pool.num_keypoints = num_keypoints;
    impl_->cuda_pool.coord_dim = impl_->coord_dim;

    impl_->cuda_pool.input_bytes =
        static_cast<size_t>(max_batch) * num_points * 3 * sizeof(float);
    impl_->cuda_pool.output_bytes =
        static_cast<size_t>(max_batch) * num_keypoints *
        static_cast<size_t>(impl_->coord_dim) * sizeof(float);

    CUDA_CHECK(cudaSetDevice(impl_->device_id));

    CUDA_CHECK(cudaMalloc(&impl_->cuda_pool.d_input, impl_->cuda_pool.input_bytes));
    CUDA_CHECK(cudaMalloc(&impl_->cuda_pool.d_output, impl_->cuda_pool.output_bytes));

    CUDA_CHECK(cudaMemset(impl_->cuda_pool.d_input, 0, impl_->cuda_pool.input_bytes));
    CUDA_CHECK(cudaMemset(impl_->cuda_pool.d_output, 0, impl_->cuda_pool.output_bytes));

    CUDA_CHECK(cudaHostAlloc(&impl_->cuda_pool.h_pinned_input,
                             impl_->cuda_pool.input_bytes,
                             cudaHostAllocPortable | cudaHostAllocWriteCombined));
    CUDA_CHECK(cudaHostAlloc(&impl_->cuda_pool.h_pinned_output,
                             impl_->cuda_pool.output_bytes,
                             cudaHostAllocPortable));

    CUDA_CHECK(cudaStreamCreateWithFlags(&impl_->cuda_stream, cudaStreamNonBlocking));

    impl_->cuda_memory_info = Ort::MemoryInfo::CreateCuda(
        OrtAllocatorType::OrtDeviceAllocator,
        OrtMemType::OrtMemTypeDefault,
        impl_->device_id);

    impl_->cpu_pinned_memory_info = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator,
        OrtMemType::OrtMemTypeCPUOutput);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        freeStaticMemoryPool();
        std::cerr << "CUDA pool allocation failed: "
                  << cudaGetErrorString(err) << std::endl;
        return false;
    }

    impl_->cuda_pool_allocated = true;

    std::cout << "[PointNet++] CUDA Memory Pool allocated:" << std::endl
              << "  max_batch = " << max_batch << std::endl
              << "  input   = " << impl_->cuda_pool.input_bytes / (1024.0 * 1024.0)
              << " MB (d_input + h_pinned_input)" << std::endl
              << "  output  = " << impl_->cuda_pool.output_bytes / (1024.0 * 1024.0)
              << " MB (d_output + h_pinned_output)" << std::endl
              << "  total   = "
              << (impl_->cuda_pool.input_bytes + impl_->cuda_pool.output_bytes)
                     * 2 / (1024.0 * 1024.0)
              << " MB" << std::endl;

    return true;
}

void PointNetInferenceEngine::freeStaticMemoryPool() {
    if (!impl_->cuda_pool_allocated) return;

    if (impl_->cuda_stream != nullptr) {
        CUDA_CHECK(cudaStreamSynchronize(impl_->cuda_stream));
    }

    if (impl_->cuda_pool.d_input) {
        CUDA_CHECK(cudaFree(impl_->cuda_pool.d_input));
        impl_->cuda_pool.d_input = nullptr;
    }
    if (impl_->cuda_pool.d_output) {
        CUDA_CHECK(cudaFree(impl_->cuda_pool.d_output));
        impl_->cuda_pool.d_output = nullptr;
    }
    if (impl_->cuda_pool.h_pinned_input) {
        CUDA_CHECK(cudaFreeHost(impl_->cuda_pool.h_pinned_input));
        impl_->cuda_pool.h_pinned_input = nullptr;
    }
    if (impl_->cuda_pool.h_pinned_output) {
        CUDA_CHECK(cudaFreeHost(impl_->cuda_pool.h_pinned_output));
        impl_->cuda_pool.h_pinned_output = nullptr;
    }

    impl_->cuda_pool_allocated = false;
    std::cout << "[PointNet++] CUDA Memory Pool released." << std::endl;
}

bool PointNetInferenceEngine::releaseCachedMemory() {
    std::lock_guard<std::mutex> lock(impl_->inference_mutex);
    if (!impl_->model_loaded) return false;

    synchronizeCuda();

#if defined(USE_TENSORRT) && (ORT_VERSION >= 11400)
    try {
        auto allocator = impl_->session->GetAllocator(0, OrtMemTypeDefault);
    } catch (...) {}
#endif

    CUDA_CHECK(cudaDeviceSynchronize());
    return true;
}

float PointNetInferenceEngine::getCurrentGpuMemoryUsedMb() const {
    size_t free = 0, total = 0;
    CUDA_CHECK(cudaSetDevice(impl_->device_id));
    cudaError_t err = cudaMemGetInfo(&free, &total);
    if (err != cudaSuccess) return 0.0f;
    return static_cast<float>(total - free) / (1024.0f * 1024.0f);
}

bool PointNetInferenceEngine::checkOomProtection() {
    float used_mb = getCurrentGpuMemoryUsedMb();
    if (used_mb > impl_->oom_threshold_mb) {
        std::cerr << "[OOM-FUSE] GPU memory " << used_mb
                  << " MB exceeds threshold " << impl_->oom_threshold_mb
                  << " MB. Triggering emergency cache purge." << std::endl;

        synchronizeCuda();
        releaseCachedMemory();
        synchronizeCuda();

        float after_mb = getCurrentGpuMemoryUsedMb();
        if (after_mb > impl_->oom_threshold_mb * 0.95f) {
            std::cerr << "[OOM-FUSE] Purge insufficient. Used after purge: "
                      << after_mb << " MB. Blocking inference." << std::endl;
            return false;
        }
    }
    return true;
}

bool PointNetInferenceEngine::loadModel(
    const std::string& onnx_model_path,
    bool use_tensorrt,
    int device_id,
    int max_batch_size) {

    try {
        CUDA_CHECK(cudaSetDevice(device_id));
        impl_->device_id = device_id;
        impl_->max_batch_size = std::min(MAX_BATCH_SIZE, std::max(1, max_batch_size));

        impl_->env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "swine3d_pointnet");
        impl_->session_options = Ort::SessionOptions();
        impl_->session_options.SetIntraOpNumThreads(impl_->intra_threads);
        impl_->session_options.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_ALL);
        impl_->session_options.SetMemPattern(true);
        impl_->session_options.EnableCpuMemArena();

#ifdef USE_TENSORRT
        if (use_tensorrt) {
            OrtTensorRTProviderOptions trt_options{};
            trt_options.device_id = device_id;
            trt_options.trt_max_workspace_size = 1LL << 28;
            trt_options.trt_fp16_enable = 1;
            trt_options.trt_engine_cache_enable = 1;
            trt_options.trt_int8_enable = 0;

            trt_options.trt_min_subgraph_size = 1;
            trt_options.trt_max_partition_iterations = 10;

            impl_->session_options.AppendExecutionProvider_TensorRT(trt_options);
            impl_->use_trt = true;
        }
#endif

        OrtCUDAProviderOptions cuda_options{};
        cuda_options.device_id = device_id;
        cuda_options.arena_extend_strategy = 0;
        cuda_options.gpu_mem_limit = 1LL << 32;
        cuda_options.do_copy_in_default_stream = 0;
        impl_->session_options.AppendExecutionProvider_CUDA(cuda_options);

        impl_->session = std::make_unique<Ort::Session>(
            impl_->env, onnx_model_path.c_str(), impl_->session_options);

        size_t num_inputs = impl_->session->GetInputCount();
        size_t num_outputs = impl_->session->GetOutputCount();

        impl_->input_names_str.clear();
        impl_->input_names.clear();
        for (size_t i = 0; i < num_inputs; ++i) {
            auto name = impl_->session->GetInputNameAllocated(i, impl_->allocator);
            impl_->input_names_str.emplace_back(name.get());
        }
        for (const auto& s : impl_->input_names_str) {
            impl_->input_names.push_back(s.c_str());
        }

        impl_->output_names_str.clear();
        impl_->output_names.clear();
        for (size_t i = 0; i < num_outputs; ++i) {
            auto name = impl_->session->GetOutputNameAllocated(i, impl_->allocator);
            impl_->output_names_str.emplace_back(name.get());
        }
        for (const auto& s : impl_->output_names_str) {
            impl_->output_names.push_back(s.c_str());
        }

        if (num_inputs > 0) {
            Ort::TypeInfo type_info = impl_->session->GetInputTypeInfo(0);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            impl_->input_shape = tensor_info.GetShape();
            if (impl_->input_shape.size() >= 2 && impl_->input_shape[1] > 0) {
                impl_->expected_point_count = static_cast<int>(impl_->input_shape[1]);
            }
        }

        if (num_outputs > 0) {
            Ort::TypeInfo type_info = impl_->session->GetOutputTypeInfo(0);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            impl_->output_shape = tensor_info.GetShape();
            if (impl_->output_shape.size() >= 2 && impl_->output_shape[1] > 0) {
                impl_->keypoint_count = static_cast<int>(impl_->output_shape[1]);
            }
            if (impl_->output_shape.size() >= 3 && impl_->output_shape[2] > 0) {
                impl_->coord_dim = static_cast<int>(impl_->output_shape[2]);
            }
        }

        if (!allocateStaticMemoryPool(
                impl_->max_batch_size,
                impl_->expected_point_count,
                impl_->keypoint_count)) {
            std::cerr << "Failed to allocate CUDA memory pool." << std::endl;
            impl_->model_loaded = false;
            return false;
        }

        impl_->model_loaded = true;
        return true;

    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime Error: " << e.what() << std::endl;
        freeStaticMemoryPool();
        impl_->model_loaded = false;
        return false;
    } catch (const std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        freeStaticMemoryPool();
        impl_->model_loaded = false;
        return false;
    }
}

bool PointNetInferenceEngine::isLoaded() const {
    return impl_->model_loaded;
}

int PointNetInferenceEngine::getExpectedPointCount() const {
    return impl_->expected_point_count;
}

int PointNetInferenceEngine::getKeypointCount() const {
    return impl_->keypoint_count;
}

int PointNetInferenceEngine::getMaxBatchSize() const {
    return impl_->max_batch_size;
}

bool PointNetInferenceEngine::validateSingleInput(
    const std::vector<Eigen::Vector3f>& point_cloud) const {
    return !point_cloud.empty() &&
           static_cast<int>(point_cloud.size()) >= impl_->expected_point_count;
}

bool PointNetInferenceEngine::validateBatchInput(const BatchPointCloud& batch) const {
    if (batch.empty() || static_cast<int>(batch.size()) > impl_->max_batch_size) {
        return false;
    }
    for (const auto& pc : batch) {
        if (!validateSingleInput(pc)) return false;
    }
    return true;
}

void PointNetInferenceEngine::copyBatchToDevice(
    const BatchPointCloud& batch, int effective_batch) {

    const int N = impl_->expected_point_count;
    const size_t stride = impl_->cuda_pool.inputStrideBytes();

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int b = 0; b < effective_batch; ++b) {
        float* dst = impl_->cuda_pool.h_pinned_input
                   + static_cast<size_t>(b) * N * 3;
        const auto& pc = batch[b];
        for (int i = 0; i < N; ++i) {
            size_t base = static_cast<size_t>(i) * 3;
            dst[base + 0] = pc[i].x();
            dst[base + 1] = pc[i].y();
            dst[base + 2] = pc[i].z();
        }
    }

    const size_t copy_bytes = static_cast<size_t>(effective_batch) * stride;
    CUDA_CHECK(cudaMemcpyAsync(
        impl_->cuda_pool.d_input,
        impl_->cuda_pool.h_pinned_input,
        copy_bytes,
        cudaMemcpyHostToDevice,
        impl_->cuda_stream));

    auto t1 = std::chrono::high_resolution_clock::now();
    impl_->last_h2d_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
}

void PointNetInferenceEngine::copyBatchFromDevice(
    BatchKeypoints& out_keypoints,
    BatchConfidences& out_confidences,
    int effective_batch) {

    const int K = impl_->keypoint_count;
    const int CD = impl_->coord_dim;
    const size_t stride = impl_->cuda_pool.outputStrideBytes();
    const bool has_conf = (CD >= 4);
    const int coords = std::min(3, CD);

    auto t0 = std::chrono::high_resolution_clock::now();

    const size_t copy_bytes = static_cast<size_t>(effective_batch) * stride;
    CUDA_CHECK(cudaMemcpyAsync(
        impl_->cuda_pool.h_pinned_output,
        impl_->cuda_pool.d_output,
        copy_bytes,
        cudaMemcpyDeviceToHost,
        impl_->cuda_stream));

    CUDA_CHECK(cudaStreamSynchronize(impl_->cuda_stream));

    auto t1 = std::chrono::high_resolution_clock::now();
    impl_->last_d2h_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

    out_keypoints.resize(static_cast<size_t>(effective_batch));
    out_confidences.resize(static_cast<size_t>(effective_batch));

    for (int b = 0; b < effective_batch; ++b) {
        const float* src = impl_->cuda_pool.h_pinned_output
                         + static_cast<size_t>(b) * K * CD;
        auto& kps = out_keypoints[b];
        auto& confs = out_confidences[b];
        kps.clear();
        confs.clear();
        kps.reserve(static_cast<size_t>(K));
        confs.reserve(static_cast<size_t>(K));

        for (int k = 0; k < K; ++k) {
            Eigen::Vector3f pt(0, 0, 0);
            for (int c = 0; c < coords; ++c) {
                pt[c] = src[k * CD + c];
            }
            kps.push_back(pt);
            float conf = has_conf ? src[k * CD + 3] : 1.0f;
            confs.push_back(conf);
        }
    }
}

std::vector<Eigen::Vector3f> PointNetInferenceEngine::infer(
    const std::vector<Eigen::Vector3f>& point_cloud) {
    std::vector<float> confs;
    return inferWithConfidence(point_cloud, confs);
}

std::vector<Eigen::Vector3f> PointNetInferenceEngine::inferWithConfidence(
    const std::vector<Eigen::Vector3f>& point_cloud,
    std::vector<float>& out_confidences) {

    BatchPointCloud batch(1, point_cloud);
    BatchConfidences batch_confs;
    BatchKeypoints batch_kps = inferBatchWithConfidence(batch, batch_confs);

    if (!batch_kps.empty()) {
        out_confidences = batch_confs[0];
        return batch_kps[0];
    }
    out_confidences.clear();
    return {};
}

PointNetInferenceEngine::BatchKeypoints PointNetInferenceEngine::inferBatch(
    const BatchPointCloud& batch_point_cloud) {
    BatchConfidences confs;
    return inferBatchWithConfidence(batch_point_cloud, confs);
}

PointNetInferenceEngine::BatchKeypoints
PointNetInferenceEngine::inferBatchWithConfidence(
    const BatchPointCloud& batch_point_cloud,
    BatchConfidences& out_batch_confidences) {

    BatchKeypoints result;
    if (!impl_->model_loaded) return result;

    std::lock_guard<std::mutex> lock(impl_->inference_mutex);

    if (impl_->shutdown_requested.load()) return result;
    if (!validateBatchInput(batch_point_cloud)) return result;
    if (!checkOomProtection()) return result;

    const int effective_batch = static_cast<int>(batch_point_cloud.size());
    const int N = impl_->expected_point_count;
    const int K = impl_->keypoint_count;

    auto t0 = std::chrono::high_resolution_clock::now();

    copyBatchToDevice(batch_point_cloud, effective_batch);

    impl_->current_input_shape = {effective_batch, N, 3};
    impl_->current_output_shape = {effective_batch, K, impl_->coord_dim};

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        impl_->cuda_memory_info,
        impl_->cuda_pool.d_input,
        static_cast<size_t>(effective_batch) * N * 3,
        impl_->current_input_shape.data(),
        impl_->current_input_shape.size());

    Ort::Value output_tensor = Ort::Value::CreateTensor<float>(
        impl_->cuda_memory_info,
        impl_->cuda_pool.d_output,
        static_cast<size_t>(effective_batch) * K * impl_->coord_dim,
        impl_->current_output_shape.data(),
        impl_->current_output_shape.size());

    try {
        impl_->session->Run(
            impl_->run_options,
            impl_->input_names.data(),
            &input_tensor,
            1,
            impl_->output_names.data(),
            &output_tensor,
            1);
    } catch (const Ort::Exception& e) {
        std::cerr << "Inference error (batch=" << effective_batch
                  << "): " << e.what() << std::endl;
        synchronizeCuda();
        return result;
    }

    copyBatchFromDevice(result, out_batch_confidences, effective_batch);

    auto t1 = std::chrono::high_resolution_clock::now();
    impl_->last_inference_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

    if (impl_->profiling) {
        std::cout << "[PointNet++] Batch=" << effective_batch
                  << ", total=" << impl_->last_inference_ms << " ms"
                  << ", H2D=" << impl_->last_h2d_ms << " ms"
                  << ", D2H=" << impl_->last_d2h_ms << " ms"
                  << ", GPU=" << getCurrentGpuMemoryUsedMb() << " MB"
                  << std::endl;
    }

    return result;
}

float PointNetInferenceEngine::getLastInferenceTimeMs() const {
    return impl_->last_inference_ms;
}

float PointNetInferenceEngine::getLastH2DTimeMs() const {
    return impl_->last_h2d_ms;
}

float PointNetInferenceEngine::getLastD2HTimeMs() const {
    return impl_->last_d2h_ms;
}

void PointNetInferenceEngine::setEnableProfiling(bool enable) {
    impl_->profiling = enable;
}

void PointNetInferenceEngine::setIntraOpNumThreads(int num_threads) {
    impl_->intra_threads = std::max(1, num_threads);
}

void PointNetInferenceEngine::setOomThresholdMb(float threshold_mb) {
    impl_->oom_threshold_mb = std::max(512.0f, threshold_mb);
}

} // namespace swine3d
