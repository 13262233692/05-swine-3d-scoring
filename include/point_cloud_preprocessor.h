#pragma once

#include "common_types.h"
#include <cstdint>
#include <unordered_map>
#include <random>

namespace swine3d {

class PointCloudPreprocessor {
public:
    PointCloudPreprocessor();

    void setPassThroughParams(const PassThroughParams& params);
    const PassThroughParams& getPassThroughParams() const;

    void setVoxelGridSize(float voxel_size);
    float getVoxelGridSize() const;

    void setTargetPointCount(int count);
    int getTargetPointCount() const;

    PointCloud applyPassThrough(const PointCloud& input) const;

    PointCloud applyVoxelDownsample(const PointCloud& input) const;

    PointCloud randomSampleFixedCount(
        const PointCloud& input,
        int target_count) const;

    PointCloud preprocess(const PointCloud& input) const;

    std::vector<Eigen::Vector3f> toEigenVectors(const PointCloud& cloud) const;

    static Eigen::Vector3f computeCentroid(const PointCloud& cloud);
    static PointCloud translateCloud(
        const PointCloud& cloud,
        const Eigen::Vector3f& offset);
    static PointCloud normalizeCloud(const PointCloud& cloud);

private:
    PassThroughParams passthrough_params_;
    float voxel_size_;
    int target_point_count_;
    mutable std::mt19937 rng_;

    struct VoxelKey {
        int64_t x, y, z;
        bool operator==(const VoxelKey& other) const {
            return x == other.x && y == other.y && z == other.z;
        }
    };

    struct VoxelKeyHash {
        size_t operator()(const VoxelKey& k) const {
            size_t h1 = std::hash<int64_t>{}(k.x);
            size_t h2 = std::hash<int64_t>{}(k.y);
            size_t h3 = std::hash<int64_t>{}(k.z);
            return h1 ^ (h2 << 1) ^ (h3 << 2);
        }
    };
};

} // namespace swine3d
