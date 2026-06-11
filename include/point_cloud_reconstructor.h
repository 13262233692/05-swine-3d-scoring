#pragma once

#include "common_types.h"
#include <string>

namespace swine3d {

class PointCloudReconstructor {
public:
    PointCloudReconstructor();
    explicit PointCloudReconstructor(const CameraIntrinsics& intrinsics);

    void setIntrinsics(const CameraIntrinsics& intrinsics);
    const CameraIntrinsics& getIntrinsics() const;

    bool loadIntrinsicsFromFile(const std::string& yaml_path);

    PointCloud reconstructFromDepthRGB(
        const cv::Mat& depth_image,
        const cv::Mat& rgb_image) const;

    PointCloud reconstructFromDepth(
        const cv::Mat& depth_image) const;

    static cv::Mat readDepthImage(const std::string& path);
    static cv::Mat readRGBImage(const std::string& path);

private:
    CameraIntrinsics intrinsics_;

    inline Eigen::Vector3f backProject(
        int u, int v, float depth_meters) const;
};

} // namespace swine3d
