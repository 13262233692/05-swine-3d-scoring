#pragma once

#include "common_types.h"
#include <string>
#include <vector>
#include <memory>

namespace swine3d {

class PointNetInferenceEngine {
public:
    PointNetInferenceEngine();
    ~PointNetInferenceEngine();

    bool loadModel(const std::string& onnx_model_path,
                   bool use_tensorrt = true,
                   int device_id = 0);

    bool isLoaded() const;

    int getExpectedPointCount() const;
    int getKeypointCount() const;

    std::vector<Eigen::Vector3f> infer(
        const std::vector<Eigen::Vector3f>& point_cloud);

    std::vector<Eigen::Vector3f> inferWithConfidence(
        const std::vector<Eigen::Vector3f>& point_cloud,
        std::vector<float>& out_confidences);

    float getLastInferenceTimeMs() const;

    void setEnableProfiling(bool enable);
    void setIntraOpNumThreads(int num_threads);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    bool validateInput(const std::vector<Eigen::Vector3f>& point_cloud) const;
};

} // namespace swine3d
