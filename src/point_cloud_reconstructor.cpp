#include "point_cloud_reconstructor.h"
#include <fstream>
#include <stdexcept>

namespace swine3d {

PointCloudReconstructor::PointCloudReconstructor()
    : intrinsics_() {}

PointCloudReconstructor::PointCloudReconstructor(const CameraIntrinsics& intrinsics)
    : intrinsics_(intrinsics) {}

void PointCloudReconstructor::setIntrinsics(const CameraIntrinsics& intrinsics) {
    intrinsics_ = intrinsics;
}

const CameraIntrinsics& PointCloudReconstructor::getIntrinsics() const {
    return intrinsics_;
}

bool PointCloudReconstructor::loadIntrinsicsFromFile(const std::string& yaml_path) {
    try {
        cv::FileStorage fs(yaml_path, cv::FileStorage::READ);
        if (!fs.isOpened()) {
            return false;
        }
        fs["fx"] >> intrinsics_.fx;
        fs["fy"] >> intrinsics_.fy;
        fs["cx"] >> intrinsics_.cx;
        fs["cy"] >> intrinsics_.cy;
        fs["width"] >> intrinsics_.width;
        fs["height"] >> intrinsics_.height;
        fs["depth_scale"] >> intrinsics_.depth_scale;
        fs.release();
        return true;
    } catch (...) {
        return false;
    }
}

inline Eigen::Vector3f PointCloudReconstructor::backProject(
    int u, int v, float depth_meters) const {
    float x = (u - intrinsics_.cx) * depth_meters / intrinsics_.fx;
    float y = (v - intrinsics_.cy) * depth_meters / intrinsics_.fy;
    return Eigen::Vector3f(x, y, depth_meters);
}

PointCloud PointCloudReconstructor::reconstructFromDepthRGB(
    const cv::Mat& depth_image,
    const cv::Mat& rgb_image) const {
    PointCloud cloud;
    if (depth_image.empty() || rgb_image.empty()) {
        return cloud;
    }

    const int h = depth_image.rows;
    const int w = depth_image.cols;
    cloud.reserve(static_cast<size_t>(h) * w);

    const bool is_16u = (depth_image.type() == CV_16UC1);
    const bool is_32f = (depth_image.type() == CV_32FC1);

    for (int v = 0; v < h; ++v) {
        for (int u = 0; u < w; ++u) {
            float depth_val = 0.0f;
            if (is_16u) {
                depth_val = static_cast<float>(depth_image.at<uint16_t>(v, u))
                          * intrinsics_.depth_scale;
            } else if (is_32f) {
                depth_val = depth_image.at<float>(v, u);
            }

            if (depth_val <= 0.0f || !std::isfinite(depth_val)) {
                continue;
            }

            Eigen::Vector3f pt = backProject(u, v, depth_val);

            cv::Vec3b color;
            if (v < rgb_image.rows && u < rgb_image.cols) {
                color = rgb_image.at<cv::Vec3b>(v, u);
            }

            cloud.emplace_back(pt.x(), pt.y(), pt.z(),
                               color[2], color[1], color[0]);
        }
    }

    return cloud;
}

PointCloud PointCloudReconstructor::reconstructFromDepth(
    const cv::Mat& depth_image) const {
    cv::Mat rgb(depth_image.size(), CV_8UC3, cv::Scalar(128, 128, 128));
    return reconstructFromDepthRGB(depth_image, rgb);
}

cv::Mat PointCloudReconstructor::readDepthImage(const std::string& path) {
    cv::Mat depth = cv::imread(path, cv::IMREAD_UNCHANGED);
    if (depth.empty()) {
        return cv::Mat();
    }
    if (depth.channels() > 1) {
        cv::cvtColor(depth, depth, cv::COLOR_BGR2GRAY);
    }
    return depth;
}

cv::Mat PointCloudReconstructor::readRGBImage(const std::string& path) {
    return cv::imread(path, cv::IMREAD_COLOR);
}

} // namespace swine3d
