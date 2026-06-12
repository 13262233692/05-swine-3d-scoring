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

struct SpineAnchor6 {
    Eigen::Vector3f left_shoulder;
    Eigen::Vector3f right_shoulder;
    Eigen::Vector3f withers;
    Eigen::Vector3f back_fat_pt;
    Eigen::Vector3f cross_section;
    Eigen::Vector3f rump;
    Eigen::Vector3f left_hip;
    Eigen::Vector3f right_hip;
};

struct BSplineSurfaceResult {
    bool success;
    int num_control_points_u;
    int num_control_points_v;
    float rmse_fit;
    std::vector<float> control_point_heights;
    float surface_area_cm2;
    float spine_curvature_max;
};

struct VolumeResult {
    float torso_volume_liters;
    float backfat_volume_liters;
    float abdomen_volume_liters;
    float belly_clearance_m;
    int   num_integration_slices;
    float integration_error_pct;
};

struct BCSResult {
    float bcs_score;
    float bcs_raw;
    float fat_indicator;
    float muscle_indicator;
    float frame_size_adjustment;
    const char* body_condition_label;
};

struct AllometricParams {
    float a_coeff;
    float b_exponent;
    float density_kg_per_liter;
    float breed_correction;
    float sex_correction;

    AllometricParams()
        : a_coeff(0.895f)
        , b_exponent(1.023f)
        , density_kg_per_liter(1.045f)
        , breed_correction(1.0f)
        , sex_correction(1.0f) {}
};

struct WeightReport {
    float estimated_weight_kg;
    float weight_lower_95ci_kg;
    float weight_upper_95ci_kg;
    float confidence_pct;
    VolumeResult volume;
    BCSResult bcs;
    BodyMeasurements measurements;
    AllometricParams params_used;
    float model_error_ema_kg;
    const char* estimation_method;
};

} // namespace swine3d
