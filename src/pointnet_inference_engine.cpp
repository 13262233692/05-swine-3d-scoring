#include "pointnet_inference_engine.h"
#include <onnxruntime_cxx_api.h>
#include <chrono>
#include <stdexcept>
#include <cstring>
#include <iostream>

namespace swine3d {

struct PointNetInferenceEngine::Impl {
    Ort::Env env{nullptr};
    Ort::SessionOptions session_options{nullptr};
    std::unique_ptr<Ort::Session> session;
    Ort::AllocatorWithDefaultOptions allocator;

    std::vector<std::string> input_names_str;
    std::vector<std::string> output_names_str;
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;

    std::vector<int64_t> input_shape;
    std::vector<int64_t> output_shape;

    int expected_point_count = 2048;
    int keypoint_count = 12;
    bool model_loaded = false;
    bool use_trt = false;
    float last_inference_ms = 0.0f;
    bool profiling = false;
    int intra_threads = 4;
};

PointNetInferenceEngine::PointNetInferenceEngine()
    : impl_(std::make_unique<Impl>()) {}

PointNetInferenceEngine::~PointNetInferenceEngine() = default;

bool PointNetInferenceEngine::loadModel(
    const std::string& onnx_model_path,
    bool use_tensorrt,
    int device_id) {
    try {
        impl_->env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "swine3d_pointnet");
        impl_->session_options = Ort::SessionOptions();

        impl_->session_options.SetIntraOpNumThreads(impl_->intra_threads);
        impl_->session_options.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_ALL);

#ifdef USE_TENSORRT
        if (use_tensorrt) {
            OrtTensorRTProviderOptions trt_options{};
            trt_options.device_id = device_id;
            trt_options.trt_max_workspace_size = 1LL << 30;
            trt_options.trt_fp16_enable = 1;
            trt_options.trt_engine_cache_enable = 1;
            impl_->session_options.AppendExecutionProvider_TensorRT(trt_options);
            impl_->use_trt = true;
        }
#endif

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
            for (size_t i = 0; i < impl_->input_shape.size(); ++i) {
                if (impl_->input_shape[i] < 0) impl_->input_shape[i] = 1;
            }
            if (impl_->input_shape.size() >= 2) {
                impl_->expected_point_count = static_cast<int>(impl_->input_shape[1]);
            }
        }

        if (num_outputs > 0) {
            Ort::TypeInfo type_info = impl_->session->GetOutputTypeInfo(0);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            impl_->output_shape = tensor_info.GetShape();
            for (size_t i = 0; i < impl_->output_shape.size(); ++i) {
                if (impl_->output_shape[i] < 0) impl_->output_shape[i] = 1;
            }
            if (impl_->output_shape.size() >= 2) {
                impl_->keypoint_count = static_cast<int>(impl_->output_shape[1]);
            }
        }

        impl_->model_loaded = true;
        return true;

    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime Error: " << e.what() << std::endl;
        impl_->model_loaded = false;
        return false;
    } catch (const std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
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

bool PointNetInferenceEngine::validateInput(
    const std::vector<Eigen::Vector3f>& point_cloud) const {
    return !point_cloud.empty() &&
           static_cast<int>(point_cloud.size()) >= impl_->expected_point_count;
}

std::vector<Eigen::Vector3f> PointNetInferenceEngine::infer(
    const std::vector<Eigen::Vector3f>& point_cloud) {
    std::vector<float> confidences;
    return inferWithConfidence(point_cloud, confidences);
}

std::vector<Eigen::Vector3f> PointNetInferenceEngine::inferWithConfidence(
    const std::vector<Eigen::Vector3f>& point_cloud,
    std::vector<float>& out_confidences) {

    std::vector<Eigen::Vector3f> keypoints;
    if (!impl_->model_loaded || !validateInput(point_cloud)) {
        return keypoints;
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    try {
        const int N = impl_->expected_point_count;
        std::vector<float> input_data(static_cast<size_t>(N) * 3);

        for (int i = 0; i < N; ++i) {
            size_t idx = static_cast<size_t>(i);
            input_data[idx * 3 + 0] = point_cloud[idx].x();
            input_data[idx * 3 + 1] = point_cloud[idx].y();
            input_data[idx * 3 + 2] = point_cloud[idx].z();
        }

        std::array<int64_t, 3> input_shape_arr = {1, N, 3};
        auto memory_info = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            input_data.data(),
            input_data.size(),
            input_shape_arr.data(),
            input_shape_arr.size());

        auto output_tensors = impl_->session->Run(
            Ort::RunOptions{nullptr},
            impl_->input_names.data(),
            &input_tensor,
            1,
            impl_->output_names.data(),
            impl_->output_names.size());

        if (output_tensors.empty()) {
            return keypoints;
        }

        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        auto out_info = output_tensors[0].GetTensorTypeAndShapeInfo();
        std::vector<int64_t> out_shape = out_info.GetShape();

        int K = impl_->keypoint_count;
        if (out_shape.size() >= 2) {
            K = static_cast<int>(out_shape[1]);
        }

        int coord_dim = 3;
        if (out_shape.size() >= 3) {
            coord_dim = static_cast<int>(out_shape[2]);
        }

        bool has_confidence = (coord_dim >= 4);
        int coords = std::min(3, coord_dim);

        keypoints.reserve(static_cast<size_t>(K));
        out_confidences.clear();
        out_confidences.reserve(static_cast<size_t>(K));

        for (int k = 0; k < K; ++k) {
            Eigen::Vector3f pt(0, 0, 0);
            for (int c = 0; c < coords; ++c) {
                pt[c] = output_data[k * coord_dim + c];
            }
            keypoints.push_back(pt);

            float conf = 1.0f;
            if (has_confidence) {
                conf = output_data[k * coord_dim + 3];
            }
            out_confidences.push_back(conf);
        }

    } catch (const Ort::Exception& e) {
        std::cerr << "Inference error: " << e.what() << std::endl;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    impl_->last_inference_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

    if (impl_->profiling) {
        std::cout << "[PointNet++] Inference time: " << impl_->last_inference_ms
                  << " ms, TRT=" << (impl_->use_trt ? "ON" : "OFF") << std::endl;
    }

    return keypoints;
}

float PointNetInferenceEngine::getLastInferenceTimeMs() const {
    return impl_->last_inference_ms;
}

void PointNetInferenceEngine::setEnableProfiling(bool enable) {
    impl_->profiling = enable;
}

void PointNetInferenceEngine::setIntraOpNumThreads(int num_threads) {
    impl_->intra_threads = std::max(1, num_threads);
}

} // namespace swine3d
