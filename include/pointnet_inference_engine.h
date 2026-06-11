#pragma once

#include "common_types.h"
#include <string>
#include <vector>
#include <memory>
#include <atomic>
#include <mutex>

namespace swine3d {

class PointNetInferenceEngine {
public:
    static constexpr int MAX_BATCH_SIZE = 4;
    static constexpr int DEFAULT_POINT_COUNT = 2048;
    static constexpr int DEFAULT_KEYPOINT_COUNT = 12;
    static constexpr float DEFAULT_OOM_THRESHOLD_MB = 3500.0f;

    using BatchPointCloud = std::vector<std::vector<Eigen::Vector3f>>;
    using BatchKeypoints = std::vector<std::vector<Eigen::Vector3f>>;
    using BatchConfidences = std::vector<std::vector<float>>;

    PointNetInferenceEngine();
    ~PointNetInferenceEngine();

    bool loadModel(const std::string& onnx_model_path,
                   bool use_tensorrt = true,
                   int device_id = 0,
                   int max_batch_size = MAX_BATCH_SIZE);

    bool isLoaded() const;

    int getExpectedPointCount() const;
    int getKeypointCount() const;
    int getMaxBatchSize() const;

    std::vector<Eigen::Vector3f> infer(
        const std::vector<Eigen::Vector3f>& point_cloud);

    std::vector<Eigen::Vector3f> inferWithConfidence(
        const std::vector<Eigen::Vector3f>& point_cloud,
        std::vector<float>& out_confidences);

    BatchKeypoints inferBatch(
        const BatchPointCloud& batch_point_cloud);

    BatchKeypoints inferBatchWithConfidence(
        const BatchPointCloud& batch_point_cloud,
        BatchConfidences& out_batch_confidences);

    float getLastInferenceTimeMs() const;
    float getLastH2DTimeMs() const;
    float getLastD2HTimeMs() const;
    float getCurrentGpuMemoryUsedMb() const;

    void setEnableProfiling(bool enable);
    void setIntraOpNumThreads(int num_threads);
    void setOomThresholdMb(float threshold_mb);

    bool releaseCachedMemory();
    void shutdown();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    bool validateSingleInput(const std::vector<Eigen::Vector3f>& point_cloud) const;
    bool validateBatchInput(const BatchPointCloud& batch) const;
    bool checkOomProtection();
    void synchronizeCuda();

    bool allocateStaticMemoryPool(int max_batch, int num_points, int num_keypoints);
    void freeStaticMemoryPool();

    void copyBatchToDevice(const BatchPointCloud& batch, int effective_batch);
    void copyBatchFromDevice(BatchKeypoints& out_keypoints,
                             BatchConfidences& out_confidences,
                             int effective_batch);
};

} // namespace swine3d
