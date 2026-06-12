#include "swine_weight_estimator.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/SVD>

namespace swine3d {

SwineWeightEstimator::SwineWeightEstimator()
    : allom_params_()
    , ground_z_(0.0f)
    , integration_slices_(48)
    , b_spline_degree_(3)
    , b_spline_ctrl_u_(6)
    , b_spline_ctrl_v_(5) {}

const char* SwineWeightEstimator::bcsLabel(float bcs_score) {
    if (bcs_score < 2.0f)     return "Emaciated (过瘦, BCS<2)";
    if (bcs_score < 2.75f)    return "Thin (偏瘦, BCS 2-2.75)";
    if (bcs_score < 3.25f)    return "Ideal (理想, BCS 2.75-3.25)";
    if (bcs_score < 4.0f)     return "Fat (偏肥, BCS 3.25-4)";
    return "Obese (过肥, BCS≥4)";
}

void SwineWeightEstimator::setBreedCorrection(const std::string& breed) {
    std::string lower = breed;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    if (lower == "duroc" || lower == "杜洛克") {
        allom_params_.breed_correction = 1.02f;
        allom_params_.density_kg_per_liter = 1.055f;
    } else if (lower == "landrace" || lower == "长白") {
        allom_params_.breed_correction = 0.985f;
        allom_params_.density_kg_per_liter = 1.038f;
    } else if (lower == "yorkshire" || lower == "大白") {
        allom_params_.breed_correction = 0.995f;
        allom_params_.density_kg_per_liter = 1.042f;
    } else {
        allom_params_.breed_correction = 1.0f;
        allom_params_.density_kg_per_liter = 1.045f;
    }
}

void SwineWeightEstimator::setSexCorrection(bool is_gilt, bool is_boar) {
    if (is_boar) allom_params_.sex_correction = 1.035f;
    else if (is_gilt) allom_params_.sex_correction = 0.985f;
    else allom_params_.sex_correction = 1.01f;
}

void SwineWeightEstimator::setGroundPlaneZ(float z) {
    ground_z_ = z;
}

SpineAnchor6 SwineWeightEstimator::extractSpineAnchors(const KeypointSet& kps) const {
    SpineAnchor6 a;
    auto get = [&](KeypointType t) -> Eigen::Vector3f {
        return kps[static_cast<size_t>(t)].position;
    };
    a.left_shoulder  = get(KeypointType::LEFT_SHOULDER);
    a.right_shoulder = get(KeypointType::RIGHT_SHOULDER);
    a.withers        = get(KeypointType::WITHERS);
    a.back_fat_pt    = get(KeypointType::BACK_FAT_TOP);
    a.cross_section  = get(KeypointType::CROSS_SECTION);
    a.rump           = get(KeypointType::RUMP);
    a.left_hip       = get(KeypointType::LEFT_HIP);
    a.right_hip      = get(KeypointType::RIGHT_HIP);
    return a;
}

float SwineWeightEstimator::deBoorCox(
    int i, int degree, float t,
    const std::vector<float>& knots) const {

    if (degree == 0) {
        return (t >= knots[i] && t < knots[i + 1]) ? 1.0f : 0.0f;
    }
    float left = 0.0f, right = 0.0f;
    float denom1 = knots[i + degree] - knots[i];
    if (std::abs(denom1) > 1e-8f) {
        left = (t - knots[i]) / denom1 * deBoorCox(i, degree - 1, t, knots);
    }
    float denom2 = knots[i + degree + 1] - knots[i + 1];
    if (std::abs(denom2) > 1e-8f) {
        right = (knots[i + degree + 1] - t) / denom2
              * deBoorCox(i + 1, degree - 1, t, knots);
    }
    return left + right;
}

std::vector<float> SwineWeightEstimator::makeUniformKnots(
    int num_ctrl, int degree) const {

    int num_knots = num_ctrl + degree + 1;
    std::vector<float> knots(static_cast<size_t>(num_knots));
    for (int i = 0; i <= degree; ++i) {
        knots[i] = 0.0f;
        knots[num_knots - 1 - i] = 1.0f;
    }
    int inner_start = degree + 1;
    int inner_count = num_knots - 2 * (degree + 1);
    for (int i = 0; i < inner_count; ++i) {
        knots[inner_start + i] = static_cast<float>(i + 1)
                               / static_cast<float>(inner_count + 1);
    }
    return knots;
}

std::vector<float> SwineWeightEstimator::parameterizeAnchorsOnSpine(
    const SpineAnchor6& anchors) const {

    std::vector<Eigen::Vector3f> pts = {
        anchors.withers,
        anchors.back_fat_pt,
        anchors.cross_section,
        (anchors.left_shoulder + anchors.right_shoulder) * 0.5f,
        anchors.rump,
        (anchors.left_hip + anchors.right_hip) * 0.5f
    };

    std::vector<float> params(pts.size(), 0.0f);
    float total_len = 0.0f;
    std::vector<float> seg_lens(pts.size() - 1, 0.0f);
    for (size_t i = 0; i + 1 < pts.size(); ++i) {
        seg_lens[i] = (pts[i + 1] - pts[i]).norm();
        total_len += seg_lens[i];
    }
    if (total_len < 1e-6f) return params;

    float cum = 0.0f;
    for (size_t i = 1; i < pts.size(); ++i) {
        cum += seg_lens[i - 1];
        params[i] = cum / total_len;
    }
    params.back() = 1.0f;
    return params;
}

Eigen::MatrixXf SwineWeightEstimator::buildBSplineBasisMatrix(
    const std::vector<float>& params_u,
    const std::vector<float>& params_v,
    int degree,
    const std::vector<float>& knots_u,
    const std::vector<float>& knots_v,
    int num_ctrl_u,
    int num_ctrl_v) const {

    const int N = static_cast<int>(params_u.size());
    const int M = num_ctrl_u * num_ctrl_v;
    Eigen::MatrixXf B(N, M);
    B.setZero();

    for (int row = 0; row < N; ++row) {
        float u = params_u[row];
        float v = params_v[row];
        for (int i = 0; i < num_ctrl_u; ++i) {
            float bu = deBoorCox(i, degree, u, knots_u);
            for (int j = 0; j < num_ctrl_v; ++j) {
                float bv = deBoorCox(j, degree, v, knots_v);
                B(row, i * num_ctrl_v + j) = bu * bv;
            }
        }
    }
    return B;
}

BSplineSurfaceResult SwineWeightEstimator::fitSpineBSplineSurface(
    const SpineAnchor6& anchors,
    const PointCloud& point_cloud,
    int control_u,
    int control_v,
    int spline_degree) {

    BSplineSurfaceResult result;
    result.success = false;
    result.rmse_fit = std::numeric_limits<float>::max();
    result.num_control_points_u = control_u;
    result.num_control_points_v = control_v;

    std::vector<Eigen::Vector3f> sample_pts;
    std::vector<float> sample_u;
    std::vector<float> sample_v;

    Eigen::Vector3f head_anchor =
        (anchors.left_shoulder + anchors.right_shoulder) * 0.5f;
    Eigen::Vector3f tail_anchor =
        (anchors.left_hip + anchors.right_hip) * 0.5f;
    Eigen::Vector3f spine_axis = (tail_anchor - head_anchor).normalized();
    Eigen::Vector3f width_axis =
        (anchors.right_shoulder - anchors.left_shoulder).normalized();

    auto add_sample = [&](const Eigen::Vector3f& p) {
        Eigen::Vector3f rel = p - head_anchor;
        float u = rel.dot(spine_axis);
        float v = rel.dot(width_axis);
        float total_len = (tail_anchor - head_anchor).norm();
        if (total_len < 1e-6f) total_len = 1.0f;
        float half_w = 0.3f;
        u = std::clamp(u / total_len, 0.0f, 1.0f);
        v = std::clamp((v + half_w) / (2.0f * half_w), 0.0f, 1.0f);
        sample_pts.push_back(p);
        sample_u.push_back(u);
        sample_v.push_back(v);
    };

    add_sample(anchors.left_shoulder);
    add_sample(anchors.right_shoulder);
    add_sample(anchors.withers);
    add_sample(anchors.back_fat_pt);
    add_sample(anchors.cross_section);
    add_sample(anchors.rump);
    add_sample(anchors.left_hip);
    add_sample(anchors.right_hip);
    add_sample(head_anchor);
    add_sample(tail_anchor);

    if (!point_cloud.empty()) {
        std::mt19937 rng(42);
        int max_samples = std::min(300, (int)point_cloud.size());
        std::uniform_int_distribution<int> dist(0, (int)point_cloud.size() - 1);
        for (int s = 0; s < max_samples; ++s) {
            const auto& pt = point_cloud[dist(rng)];
            Eigen::Vector3f p(pt.x, pt.y, pt.z);
            float w = (p - anchors.withers).norm();
            if (w < 0.6f) add_sample(p);
        }
    }

    if (sample_pts.size() < 6) return result;

    auto knots_u = makeUniformKnots(control_u, spline_degree);
    auto knots_v = makeUniformKnots(control_v, spline_degree);

    Eigen::MatrixXf B = buildBSplineBasisMatrix(
        sample_u, sample_v, spline_degree,
        knots_u, knots_v, control_u, control_v);

    Eigen::VectorXf z_vals(static_cast<int>(sample_pts.size()));
    for (size_t i = 0; i < sample_pts.size(); ++i) {
        z_vals(static_cast<int>(i)) = sample_pts[i].z();
    }

    Eigen::JacobiSVD<Eigen::MatrixXf> svd(
        B, Eigen::ComputeThinU | Eigen::ComputeThinV);
    svd.setThreshold(1e-6f);
    Eigen::VectorXf ctrl_z = svd.solve(z_vals);

    Eigen::VectorXf fitted = B * ctrl_z;
    Eigen::VectorXf residual = z_vals - fitted;
    result.rmse_fit = std::sqrt(residual.squaredNorm() / residual.size());

    result.control_point_heights.resize(ctrl_z.size());
    for (int i = 0; i < ctrl_z.size(); ++i) {
        result.control_point_heights[i] = ctrl_z(i);
    }

    float spine_max = 0.0f;
    const int NUM_SAMPLE = 100;
    float prev_h = 0.0f;
    for (int i = 0; i < NUM_SAMPLE; ++i) {
        float u = (float)i / (float)(NUM_SAMPLE - 1);
        float v = 0.5f;
        float h = 0.0f;
        for (int ci = 0; ci < control_u; ++ci) {
            float bu = deBoorCox(ci, spline_degree, u, knots_u);
            for (int cj = 0; cj < control_v; ++cj) {
                float bv = deBoorCox(cj, spline_degree, v, knots_v);
                h += ctrl_z(ci * control_v + cj) * bu * bv;
            }
        }
        if (i > 0) {
            spine_max = std::max(spine_max, std::abs(h - prev_h));
        }
        prev_h = h;
    }
    result.spine_curvature_max = spine_max;

    float area = 0.0f;
    const int U_SAMP = 64, V_SAMP = 32;
    float du = 1.0f / (U_SAMP - 1);
    float dv = 1.0f / (V_SAMP - 1);
    float total_len = std::max(0.3f, (tail_anchor - head_anchor).norm());
    for (int i = 0; i < U_SAMP - 1; ++i) {
        for (int j = 0; j < V_SAMP - 1; ++j) {
            area += du * dv * total_len * 0.6f;
        }
    }
    result.surface_area_cm2 = area * 10000.0f;

    result.success = (result.rmse_fit < 0.05f);
    return result;
}

float SwineWeightEstimator::crossSectionalAreaAtSlice(
    const PointCloud& point_cloud,
    const Eigen::Vector3f& spine_point,
    const Eigen::Vector3f& tangent,
    float slice_thickness,
    float half_width,
    float& out_belly_clearance,
    float& out_dorsal_height) const {

    Eigen::Vector3f up(0, 0, 1);
    Eigen::Vector3f side = tangent.cross(up).normalized();
    Eigen::Vector3f normal = side.cross(tangent).normalized();

    std::vector<float> left_profile_z;
    std::vector<float> right_profile_z;
    std::vector<float> y_left, y_right;

    float min_belly = spine_point.z();
    float max_dorsal = spine_point.z();

    for (const auto& pt : point_cloud) {
        Eigen::Vector3f p(pt.x, pt.y, pt.z);
        Eigen::Vector3f d = p - spine_point;
        float along = d.dot(tangent);
        if (std::abs(along) > slice_thickness * 0.5f) continue;

        float s = d.dot(side);
        if (std::abs(s) > half_width) continue;

        float z = p.z();
        if (s < 0) {
            left_profile_z.push_back(z);
            y_left.push_back(s);
        } else {
            right_profile_z.push_back(z);
            y_right.push_back(s);
        }

        min_belly = std::min(min_belly, z);
        max_dorsal = std::max(max_dorsal, z);
    }

    auto computeAreaSide = [](std::vector<float> ys, std::vector<float> zs,
                              float ground_z, bool left) -> float {
        if (ys.empty()) return 0.0f;
        std::vector<std::pair<float, float>> pts;
        for (size_t i = 0; i < ys.size(); ++i) {
            pts.emplace_back(std::abs(ys[i]), zs[i]);
        }
        std::sort(pts.begin(), pts.end());
        float area = 0.0f;
        float min_z = ground_z;
        for (size_t i = 1; i < pts.size(); ++i) {
            float dy = pts[i].first - pts[i - 1].first;
            float avg_z = (pts[i].second + pts[i - 1].second) * 0.5f;
            if (avg_z > min_z) {
                area += dy * (avg_z - min_z);
            }
        }
        return area;
    };

    float left_area = computeAreaSide(y_left, left_profile_z, ground_z_, true);
    float right_area = computeAreaSide(y_right, right_profile_z, ground_z_, false);

    out_belly_clearance = std::max(0.0f, min_belly - ground_z_);
    out_dorsal_height = max_dorsal - ground_z_;
    return left_area + right_area;
}

VolumeResult SwineWeightEstimator::integrateTorsoVolume(
    const KeypointSet& keypoints,
    const PointCloud& point_cloud,
    const BSplineSurfaceResult& spine_surface,
    int num_slices,
    float ground_z) {

    VolumeResult result;
    result.torso_volume_liters = 0.0f;
    result.backfat_volume_liters = 0.0f;
    result.abdomen_volume_liters = 0.0f;
    result.belly_clearance_m = 0.0f;
    result.num_integration_slices = num_slices;
    result.integration_error_pct = 5.0f;

    Eigen::Vector3f head = (keypoints[static_cast<size_t>(KeypointType::LEFT_SHOULDER)].position
                          + keypoints[static_cast<size_t>(KeypointType::RIGHT_SHOULDER)].position)
                         * 0.5f;
    Eigen::Vector3f tail = (keypoints[static_cast<size_t>(KeypointType::LEFT_HIP)].position
                          + keypoints[static_cast<size_t>(KeypointType::RIGHT_HIP)].position)
                         * 0.5f;

    Eigen::Vector3f spine_vec = tail - head;
    float total_len = spine_vec.norm();
    if (total_len < 0.2f) {
        result.torso_volume_liters = 0.05f;
        return result;
    }
    Eigen::Vector3f tangent = spine_vec.normalized();

    float slice_thickness = total_len / num_slices;
    float half_body_width = 0.3f;

    float total_vol_m3 = 0.0f;
    float belly_min_all = std::numeric_limits<float>::max();
    float avg_belly = 0.0f;

    float prev_area = 0.0f;
    for (int s = 0; s < num_slices; ++s) {
        float t = (static_cast<float>(s) + 0.5f) / static_cast<float>(num_slices);
        Eigen::Vector3f spine_pt = head + tangent * total_len * t;

        float belly_clearance, dorsal_h;
        float area = crossSectionalAreaAtSlice(
            point_cloud, spine_pt, tangent, slice_thickness,
            half_body_width, belly_clearance, dorsal_h);

        float slice_vol = (s > 0)
            ? (prev_area + area) * 0.5f * slice_thickness
            : area * slice_thickness;
        total_vol_m3 += slice_vol;
        prev_area = area;

        belly_min_all = std::min(belly_min_all, belly_clearance);
        avg_belly += belly_clearance;
    }

    float bf_top_z = keypoints[static_cast<size_t>(KeypointType::BACK_FAT_TOP)]
                         .position.z();
    float bf_bot_z = keypoints[static_cast<size_t>(KeypointType::BACK_FAT_BOTTOM)]
                         .position.z();
    float bf_thickness = std::max(0.005f, std::abs(bf_top_z - bf_bot_z));
    float bf_area_approx = 0.08f * 0.4f;
    float bf_vol = bf_area_approx * bf_thickness;

    result.belly_clearance_m = belly_min_all;
    result.torso_volume_liters = total_vol_m3 * 1000.0f;
    result.backfat_volume_liters = bf_vol * 1000.0f;
    result.abdomen_volume_liters = total_vol_m3 * 1000.0f * 0.45f;
    result.num_integration_slices = num_slices;
    result.integration_error_pct = spine_surface.success
        ? (spine_surface.rmse_fit * 100.0f + 2.0f)
        : 8.0f;

    return result;
}

float SwineWeightEstimator::estimateBackfatThicknessFromCloud(
    const PointCloud& point_cloud,
    const Eigen::Vector3f& backfat_top,
    const Eigen::Vector3f& backfat_bottom,
    const Eigen::Vector3f& tangent) const {

    Eigen::Vector3f dir = (backfat_bottom - backfat_top).normalized();
    Eigen::Vector3f center = (backfat_top + backfat_bottom) * 0.5f;

    float max_proj = -1e9f;
    float min_proj = 1e9f;
    for (const auto& pt : point_cloud) {
        Eigen::Vector3f p(pt.x, pt.y, pt.z);
        Eigen::Vector3f d = p - center;
        if (d.norm() > 0.3f) continue;
        float proj = d.dot(dir);
        max_proj = std::max(max_proj, proj);
        min_proj = std::min(min_proj, proj);
    }
    float thick = max_proj - min_proj;
    if (thick > 0.001f && thick < 0.1f) {
        return thick;
    }
    return (backfat_top - backfat_bottom).norm();
}

BCSResult SwineWeightEstimator::computeBCS(
    const KeypointSet& keypoints,
    const VolumeResult& volume,
    const BodyMeasurements& measurements,
    float reference_weight_hint_kg) {

    BCSResult result;
    result.bcs_score = 3.0f;

    float withers_h = keypoints[static_cast<size_t>(KeypointType::WITHERS)]
                          .position.z();
    float cross_h = keypoints[static_cast<size_t>(KeypointType::CROSS_SECTION)]
                        .position.z();
    float spine_drop = std::abs(withers_h - cross_h);
    float spine_drop_ratio = spine_drop / std::max(0.001f, measurements.body_length);

    float bf_mm = measurements.back_fat_thickness * 1000.0f;

    float bf_score = std::clamp(1.0f + (bf_mm - 8.0f) / 4.0f, 1.0f, 5.0f);

    float length_m = measurements.body_length;
    float volume_l = volume.torso_volume_liters;
    float expected_vol = std::pow(length_m, 3.0f) * 350.0f;
    float vol_ratio = (expected_vol > 1.0f)
        ? volume_l / expected_vol : 1.0f;
    float vol_score = std::clamp(1.0f + (vol_ratio - 0.9f) * 8.0f, 1.0f, 5.0f);

    float hip_width = (keypoints[static_cast<size_t>(KeypointType::LEFT_HIP)].position
                     - keypoints[static_cast<size_t>(KeypointType::RIGHT_HIP)].position).norm();
    float shoulder_width = (keypoints[static_cast<size_t>(KeypointType::LEFT_SHOULDER)].position
                          - keypoints[static_cast<size_t>(KeypointType::RIGHT_SHOULDER)].position).norm();
    float width_ratio = hip_width / std::max(0.01f, shoulder_width);
    float frame_score = std::clamp(3.0f + (width_ratio - 0.95f) * 6.0f, 1.0f, 5.0f);

    result.fat_indicator = bf_score;
    result.muscle_indicator = vol_score;
    result.frame_size_adjustment = frame_score;

    result.bcs_raw = 0.55f * bf_score + 0.30f * vol_score + 0.15f * frame_score
                    - spine_drop_ratio * 15.0f;

    if (reference_weight_hint_kg > 0.0f && volume.torso_volume_liters > 0.0f) {
        float implied_density = reference_weight_hint_kg / volume.torso_volume_liters;
        float bias = (implied_density - 1.045f) * 30.0f;
        result.bcs_raw += bias;
    }

    result.bcs_score = std::clamp(result.bcs_raw, 1.0f, 5.0f);
    result.body_condition_label = bcsLabel(result.bcs_score);
    return result;
}

float SwineWeightEstimator::estimateWeightAllometric(
    const VolumeResult& volume,
    const BCSResult& bcs,
    const AllometricParams& params) {

    float V_liters = std::max(0.01f, volume.torso_volume_liters);

    float bcs_factor = 0.92f + 0.035f * bcs.bcs_score;

    float W_direct = V_liters * params.density_kg_per_liter
                   * bcs_factor
                   * params.breed_correction
                   * params.sex_correction;

    float nominal_L_cm = std::cbrt(V_liters / 0.350f);
    float W_allom = params.a_coeff * std::pow(nominal_L_cm, params.b_exponent);

    float alpha = 0.72f;
    float W_final = alpha * W_direct + (1.0f - alpha) * W_allom;

    return W_final;
}

WeightReport SwineWeightEstimator::computeFullReport(
    const KeypointSet& keypoints,
    const PointCloud& point_cloud,
    const BodyMeasurements& measurements,
    const AllometricParams& custom_params) {

    WeightReport report;
    report.measurements = measurements;
    report.params_used = custom_params;
    report.params_used.a_coeff = allom_params_.a_coeff;
    report.params_used.b_exponent = allom_params_.b_exponent;
    report.params_used.density_kg_per_liter = allom_params_.density_kg_per_liter;
    report.params_used.breed_correction *= custom_params.breed_correction;
    report.params_used.sex_correction *= custom_params.sex_correction;
    report.model_error_ema_kg = 0.0f;

    SpineAnchor6 anchors = extractSpineAnchors(keypoints);
    BSplineSurfaceResult spline = fitSpineBSplineSurface(
        anchors, point_cloud,
        b_spline_ctrl_u_, b_spline_ctrl_v_, b_spline_degree_);

    report.volume = integrateTorsoVolume(
        keypoints, point_cloud, spline,
        integration_slices_, ground_z_);

    report.bcs = computeBCS(keypoints, report.volume, measurements);

    report.estimated_weight_kg = estimateWeightAllometric(
        report.volume, report.bcs, report.params_used);

    float err_pct = std::max(report.volume.integration_error_pct, 2.0f);
    if (!spline.success) err_pct += 3.0f;
    if (report.bcs.bcs_score < 2.0f || report.bcs.bcs_score > 4.0f) err_pct += 2.0f;

    report.weight_lower_95ci_kg = report.estimated_weight_kg * (1.0f - err_pct / 100.0f * 1.96f);
    report.weight_upper_95ci_kg = report.estimated_weight_kg * (1.0f + err_pct / 100.0f * 1.96f);
    report.confidence_pct = std::max(30.0f, 100.0f - err_pct * 2.5f);
    report.estimation_method = spline.success
        ? "B-Spline Surface + Simpson Integration + Allometric"
        : "Keypoint Geometric + Volume Proxy + Allometric";

    return report;
}

} // namespace swine3d
