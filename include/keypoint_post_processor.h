#pragma once

#include "common_types.h"

namespace swine3d {

class KeypointPostProcessor {
public:
    KeypointPostProcessor();

    KeypointSet mapToWorldSpace(
        const std::vector<Eigen::Vector3f>& normalized_keypoints,
        const std::vector<float>& confidences,
        const PointCloud& original_cloud,
        const Eigen::Vector3f& centroid,
        float scale_factor) const;

    KeypointSet denormalizeKeypoints(
        const std::vector<Eigen::Vector3f>& normalized_keypoints,
        const Eigen::Vector3f& centroid,
        float scale_factor) const;

    KeypointSet createKeypointSet(
        const std::vector<Eigen::Vector3f>& positions,
        const std::vector<float>& confidences) const;

    static BodyMeasurements computeMeasurements(const KeypointSet& keypoints);

    static float computeDistance(
        const Keypoint3D& a, const Keypoint3D& b);

    Eigen::Vector3f projectToPointCloud(
        const Eigen::Vector3f& query,
        const PointCloud& cloud,
        int k_nearest = 5) const;

    void setConfidenceThreshold(float threshold);
    float getConfidenceThreshold() const;

    static const char* keypointTypeName(KeypointType type);

private:
    float confidence_threshold_;

    static float distance(const Eigen::Vector3f& a, const Eigen::Vector3f& b);
};

class KeypointSmoother {
public:
    KeypointSmoother();
    explicit KeypointSmoother(float alpha);

    KeypointSet update(const KeypointSet& current);
    void reset();
    void setAlpha(float alpha);

private:
    float alpha_;
    bool has_history_;
    KeypointSet history_;
};

} // namespace swine3d
