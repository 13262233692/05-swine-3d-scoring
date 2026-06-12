#pragma once

#include "common_types.h"

namespace swine3d {

class SwineWeightEstimator {
public:
    SwineWeightEstimator();

    WeightReport computeFullReport(
        const KeypointSet& keypoints,
        const PointCloud& point_cloud,
        const BodyMeasurements& measurements,
        const AllometricParams& custom_params = AllometricParams());

    SpineAnchor6 extractSpineAnchors(const KeypointSet& keypoints) const;

    BSplineSurfaceResult fitSpineBSplineSurface(
        const SpineAnchor6& anchors,
        const PointCloud& point_cloud,
        int control_u = 6,
        int control_v = 5,
        int spline_degree = 3);

    VolumeResult integrateTorsoVolume(
        const KeypointSet& keypoints,
        const PointCloud& point_cloud,
        const BSplineSurfaceResult& spine_surface,
        int num_slices = 48,
        float ground_z = 0.0f);

    BCSResult computeBCS(
        const KeypointSet& keypoints,
        const VolumeResult& volume,
        const BodyMeasurements& measurements,
        float reference_weight_hint_kg = 0.0f);

    float estimateWeightAllometric(
        const VolumeResult& volume,
        const BCSResult& bcs,
        const AllometricParams& params);

    void setBreedCorrection(const std::string& breed);
    void setSexCorrection(bool is_gilt, bool is_boar);
    void setGroundPlaneZ(float z);

    static const char* bcsLabel(float bcs_score);

private:
    AllometricParams allom_params_;
    float ground_z_;
    int   integration_slices_;
    int   b_spline_degree_;
    int   b_spline_ctrl_u_;
    int   b_spline_ctrl_v_;

    float deBoorCox(int i, int degree, float t,
                    const std::vector<float>& knots) const;

    Eigen::MatrixXf buildBSplineBasisMatrix(
        const std::vector<float>& params_u,
        const std::vector<float>& params_v,
        int degree,
        const std::vector<float>& knots_u,
        const std::vector<float>& knots_v,
        int num_ctrl_u,
        int num_ctrl_v) const;

    std::vector<float> makeUniformKnots(int num_ctrl, int degree) const;

    std::vector<float> parameterizeAnchorsOnSpine(
        const SpineAnchor6& anchors) const;

    float crossSectionalAreaAtSlice(
        const PointCloud& point_cloud,
        const Eigen::Vector3f& spine_point,
        const Eigen::Vector3f& tangent,
        float slice_thickness,
        float half_width,
        float& out_belly_clearance,
        float& out_dorsal_height) const;

    float estimateBackfatThicknessFromCloud(
        const PointCloud& point_cloud,
        const Eigen::Vector3f& backfat_top,
        const Eigen::Vector3f& backfat_bottom,
        const Eigen::Vector3f& tangent) const;
};

} // namespace swine3d
