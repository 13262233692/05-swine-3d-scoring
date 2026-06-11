#pragma once

#include <vector>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

namespace swine3d {

struct PointXYZRGB {
    float x;
    float y;
    float z;
    uint8_t r;
    uint8_t g;
    uint8_t b;

    PointXYZRGB() : x(0), y(0), z(0), r(0), g(0), b(0) {}
    PointXYZRGB(float x_, float y_, float z_, uint8_t r_, uint8_t g_, uint8_t b_)
        : x(x_), y(y_), z(z_), r(r_), g(g_), b(b_) {}
};

using PointCloud = std::vector<PointXYZRGB>;

enum class KeypointType {
    LEFT_EAR = 0,
    RIGHT_EAR = 1,
    LEFT_SHOULDER = 2,
    RIGHT_SHOULDER = 3,
    WITHERS = 4,
    BACK_FAT_TOP = 5,
    BACK_FAT_BOTTOM = 6,
    CROSS_SECTION = 7,
    LEFT_HIP = 8,
    RIGHT_HIP = 9,
    RUMP = 10,
    TAIL_HEAD = 11,
    NUM_KEYPOINTS = 12
};

struct Keypoint3D {
    KeypointType type;
    Eigen::Vector3f position;
    float confidence;

    Keypoint3D() : type(KeypointType::NUM_KEYPOINTS), position(0, 0, 0), confidence(0.0f) {}
    Keypoint3D(KeypointType t, const Eigen::Vector3f& p, float c)
        : type(t), position(p), confidence(c) {}
};

using KeypointSet = std::array<Keypoint3D, static_cast<size_t>(KeypointType::NUM_KEYPOINTS)>;

struct CameraIntrinsics {
    float fx;
    float fy;
    float cx;
    float cy;
    int width;
    int height;
    float depth_scale;

    CameraIntrinsics()
        : fx(640.0f), fy(640.0f), cx(320.0f), cy(240.0f),
          width(640), height(480), depth_scale(0.001f) {}
};

struct PassThroughParams {
    float x_min;
    float x_max;
    float y_min;
    float y_max;
    float z_min;
    float z_max;

    PassThroughParams()
        : x_min(-1.0f), x_max(1.0f),
          y_min(-0.5f), y_max(0.5f),
          z_min(0.3f), z_max(2.0f) {}
};

struct BodyMeasurements {
    float body_length;
    float body_width;
    float body_height;
    float back_fat_thickness;
    float chest_circumference;

    BodyMeasurements()
        : body_length(0), body_width(0), body_height(0),
          back_fat_thickness(0), chest_circumference(0) {}
};

} // namespace swine3d
