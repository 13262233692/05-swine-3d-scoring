#include "point_cloud_preprocessor.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace swine3d {

PointCloudPreprocessor::PointCloudPreprocessor()
    : passthrough_params_()
    , voxel_size_(0.005f)
    , target_point_count_(2048)
    , rng_(42) {}

void PointCloudPreprocessor::setPassThroughParams(const PassThroughParams& params) {
    passthrough_params_ = params;
}

const PassThroughParams& PointCloudPreprocessor::getPassThroughParams() const {
    return passthrough_params_;
}

void PointCloudPreprocessor::setVoxelGridSize(float voxel_size) {
    voxel_size_ = std::max(1e-6f, voxel_size);
}

float PointCloudPreprocessor::getVoxelGridSize() const {
    return voxel_size_;
}

void PointCloudPreprocessor::setTargetPointCount(int count) {
    target_point_count_ = std::max(1, count);
}

int PointCloudPreprocessor::getTargetPointCount() const {
    return target_point_count_;
}

PointCloud PointCloudPreprocessor::applyPassThrough(const PointCloud& input) const {
    PointCloud output;
    output.reserve(input.size());

    const auto& p = passthrough_params_;
    for (const auto& pt : input) {
        if (pt.x >= p.x_min && pt.x <= p.x_max &&
            pt.y >= p.y_min && pt.y <= p.y_max &&
            pt.z >= p.z_min && pt.z <= p.z_max) {
            output.push_back(pt);
        }
    }

    return output;
}

PointCloud PointCloudPreprocessor::applyVoxelDownsample(const PointCloud& input) const {
    if (input.empty() || voxel_size_ <= 0.0f) {
        return input;
    }

    std::unordered_map<VoxelKey, std::pair<PointXYZRGB, size_t>, VoxelKeyHash> voxel_map;

    for (const auto& pt : input) {
        VoxelKey key;
        key.x = static_cast<int64_t>(std::floor(pt.x / voxel_size_));
        key.y = static_cast<int64_t>(std::floor(pt.y / voxel_size_));
        key.z = static_cast<int64_t>(std::floor(pt.z / voxel_size_));

        auto& entry = voxel_map[key];
        entry.first.x += pt.x;
        entry.first.y += pt.y;
        entry.first.z += pt.z;
        entry.first.r += pt.r;
        entry.first.g += pt.g;
        entry.first.b += pt.b;
        entry.second++;
    }

    PointCloud output;
    output.reserve(voxel_map.size());

    for (const auto& kv : voxel_map) {
        const auto& entry = kv.second;
        const size_t n = entry.second;
        const float inv_n = 1.0f / static_cast<float>(n);
        PointXYZRGB avg;
        avg.x = entry.first.x * inv_n;
        avg.y = entry.first.y * inv_n;
        avg.z = entry.first.z * inv_n;
        avg.r = static_cast<uint8_t>(std::min(255, static_cast<int>(entry.first.r * inv_n)));
        avg.g = static_cast<uint8_t>(std::min(255, static_cast<int>(entry.first.g * inv_n)));
        avg.b = static_cast<uint8_t>(std::min(255, static_cast<int>(entry.first.b * inv_n)));
        output.push_back(avg);
    }

    return output;
}

PointCloud PointCloudPreprocessor::randomSampleFixedCount(
    const PointCloud& input, int target_count) const {
    if (input.size() <= static_cast<size_t>(target_count)) {
        PointCloud output = input;
        while (output.size() < static_cast<size_t>(target_count) && !input.empty()) {
            std::uniform_int_distribution<size_t> dist(0, input.size() - 1);
            output.push_back(input[dist(rng_)]);
        }
        return output;
    }

    PointCloud shuffled = input;
    std::shuffle(shuffled.begin(), shuffled.end(), rng_);
    shuffled.resize(static_cast<size_t>(target_count));
    return shuffled;
}

PointCloud PointCloudPreprocessor::preprocess(const PointCloud& input) const {
    PointCloud filtered = applyPassThrough(input);
    PointCloud downsampled = applyVoxelDownsample(filtered);
    PointCloud sampled = randomSampleFixedCount(downsampled, target_point_count_);
    return normalizeCloud(sampled);
}

std::vector<Eigen::Vector3f> PointCloudPreprocessor::toEigenVectors(
    const PointCloud& cloud) const {
    std::vector<Eigen::Vector3f> vectors;
    vectors.reserve(cloud.size());
    for (const auto& pt : cloud) {
        vectors.emplace_back(pt.x, pt.y, pt.z);
    }
    return vectors;
}

Eigen::Vector3f PointCloudPreprocessor::computeCentroid(const PointCloud& cloud) {
    Eigen::Vector3f centroid(0, 0, 0);
    if (cloud.empty()) return centroid;

    for (const auto& pt : cloud) {
        centroid.x() += pt.x;
        centroid.y() += pt.y;
        centroid.z() += pt.z;
    }
    centroid /= static_cast<float>(cloud.size());
    return centroid;
}

PointCloud PointCloudPreprocessor::translateCloud(
    const PointCloud& cloud, const Eigen::Vector3f& offset) {
    PointCloud result = cloud;
    for (auto& pt : result) {
        pt.x += offset.x();
        pt.y += offset.y();
        pt.z += offset.z();
    }
    return result;
}

PointCloud PointCloudPreprocessor::normalizeCloud(const PointCloud& cloud) {
    if (cloud.empty()) return cloud;

    Eigen::Vector3f centroid = computeCentroid(cloud);

    float max_dist = 0.0f;
    for (const auto& pt : cloud) {
        float dx = pt.x - centroid.x();
        float dy = pt.y - centroid.y();
        float dz = pt.z - centroid.z();
        float dist = dx * dx + dy * dy + dz * dz;
        max_dist = std::max(max_dist, dist);
    }
    max_dist = std::sqrt(max_dist);
    if (max_dist < 1e-6f) max_dist = 1.0f;

    PointCloud result = translateCloud(cloud, -centroid);
    float scale = 1.0f / max_dist;
    for (auto& pt : result) {
        pt.x *= scale;
        pt.y *= scale;
        pt.z *= scale;
    }
    return result;
}

} // namespace swine3d
