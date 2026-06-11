#include "keypoint_post_processor.h"
#include <algorithm>
#include <cmath>
#include <queue>
#include <limits>

namespace swine3d {

KeypointPostProcessor::KeypointPostProcessor()
    : confidence_threshold_(0.3f) {}

void KeypointPostProcessor::setConfidenceThreshold(float threshold) {
    confidence_threshold_ = threshold;
}

float KeypointPostProcessor::getConfidenceThreshold() const {
    return confidence_threshold_;
}

const char* KeypointPostProcessor::keypointTypeName(KeypointType type) {
    switch (type) {
        case KeypointType::LEFT_EAR:        return "LeftEar";
        case KeypointType::RIGHT_EAR:       return "RightEar";
        case KeypointType::LEFT_SHOULDER:   return "LeftShoulder";
        case KeypointType::RIGHT_SHOULDER:  return "RightShoulder";
        case KeypointType::WITHERS:         return "Withers";
        case KeypointType::BACK_FAT_TOP:    return "BackFatTop";
        case KeypointType::BACK_FAT_BOTTOM: return "BackFatBottom";
        case KeypointType::CROSS_SECTION:   return "CrossSection";
        case KeypointType::LEFT_HIP:        return "LeftHip";
        case KeypointType::RIGHT_HIP:       return "RightHip";
        case KeypointType::RUMP:            return "Rump";
        case KeypointType::TAIL_HEAD:       return "TailHead";
        default:                            return "Unknown";
    }
}

float KeypointPostProcessor::distance(
    const Eigen::Vector3f& a, const Eigen::Vector3f& b) {
    return (a - b).norm();
}

KeypointSet KeypointPostProcessor::createKeypointSet(
    const std::vector<Eigen::Vector3f>& positions,
    const std::vector<float>& confidences) const {

    KeypointSet set;
    size_t K = static_cast<size_t>(KeypointType::NUM_KEYPOINTS);

    for (size_t i = 0; i < K; ++i) {
        Keypoint3D kp;
        kp.type = static_cast<KeypointType>(i);
        if (i < positions.size()) {
            kp.position = positions[i];
        }
        if (i < confidences.size()) {
            kp.confidence = confidences[i];
        }
        set[i] = kp;
    }
    return set;
}

KeypointSet KeypointPostProcessor::denormalizeKeypoints(
    const std::vector<Eigen::Vector3f>& normalized_keypoints,
    const Eigen::Vector3f& centroid,
    float scale_factor) const {

    KeypointSet set;
    size_t K = static_cast<size_t>(KeypointType::NUM_KEYPOINTS);

    for (size_t i = 0; i < K; ++i) {
        Keypoint3D kp;
        kp.type = static_cast<KeypointType>(i);
        kp.confidence = 1.0f;
        if (i < normalized_keypoints.size()) {
            kp.position = normalized_keypoints[i] * scale_factor + centroid;
        }
        set[i] = kp;
    }
    return set;
}

Eigen::Vector3f KeypointPostProcessor::projectToPointCloud(
    const Eigen::Vector3f& query,
    const PointCloud& cloud,
    int k_nearest) const {

    if (cloud.empty()) return query;

    std::priority_queue<std::pair<float, size_t>> pq;

    for (size_t i = 0; i < cloud.size(); ++i) {
        const auto& pt = cloud[i];
        Eigen::Vector3f p(pt.x, pt.y, pt.z);
        float d = (p - query).squaredNorm();
        if (pq.size() < static_cast<size_t>(k_nearest)) {
            pq.push({d, i});
        } else if (d < pq.top().first) {
            pq.pop();
            pq.push({d, i});
        }
    }

    Eigen::Vector3f avg(0, 0, 0);
    int count = 0;
    while (!pq.empty()) {
        size_t idx = pq.top().second;
        pq.pop();
        avg.x() += cloud[idx].x;
        avg.y() += cloud[idx].y;
        avg.z() += cloud[idx].z;
        ++count;
    }
    if (count > 0) {
        avg /= static_cast<float>(count);
    }
    return (count > 0) ? avg : query;
}

KeypointSet KeypointPostProcessor::mapToWorldSpace(
    const std::vector<Eigen::Vector3f>& normalized_keypoints,
    const std::vector<float>& confidences,
    const PointCloud& original_cloud,
    const Eigen::Vector3f& centroid,
    float scale_factor) const {

    KeypointSet set = createKeypointSet(normalized_keypoints, confidences);

    for (auto& kp : set) {
        kp.position = kp.position * scale_factor + centroid;
        if (kp.confidence >= confidence_threshold_ && !original_cloud.empty()) {
            kp.position = projectToPointCloud(kp.position, original_cloud, 3);
        }
    }
    return set;
}

float KeypointPostProcessor::computeDistance(
    const Keypoint3D& a, const Keypoint3D& b) {
    return distance(a.position, b.position);
}

BodyMeasurements KeypointPostProcessor::computeMeasurements(
    const KeypointSet& keypoints) {

    BodyMeasurements m;

    const auto& withers = keypoints[static_cast<size_t>(KeypointType::WITHERS)];
    const auto& tail_head = keypoints[static_cast<size_t>(KeypointType::TAIL_HEAD)];
    const auto& left_shoulder = keypoints[static_cast<size_t>(KeypointType::LEFT_SHOULDER)];
    const auto& right_shoulder = keypoints[static_cast<size_t>(KeypointType::RIGHT_SHOULDER)];
    const auto& left_hip = keypoints[static_cast<size_t>(KeypointType::LEFT_HIP)];
    const auto& right_hip = keypoints[static_cast<size_t>(KeypointType::RIGHT_HIP)];
    const auto& bf_top = keypoints[static_cast<size_t>(KeypointType::BACK_FAT_TOP)];
    const auto& bf_bottom = keypoints[static_cast<size_t>(KeypointType::BACK_FAT_BOTTOM)];
    const auto& cross = keypoints[static_cast<size_t>(KeypointType::CROSS_SECTION)];

    if (withers.confidence > 0.3f && tail_head.confidence > 0.3f) {
        m.body_length = computeDistance(withers, tail_head);
    }

    float shoulder_w = 0.0f, hip_w = 0.0f;
    int count = 0;
    if (left_shoulder.confidence > 0.3f && right_shoulder.confidence > 0.3f) {
        shoulder_w = computeDistance(left_shoulder, right_shoulder);
        ++count;
    }
    if (left_hip.confidence > 0.3f && right_hip.confidence > 0.3f) {
        hip_w = computeDistance(left_hip, right_hip);
        ++count;
    }
    if (count > 0) {
        m.body_width = (shoulder_w + hip_w) / static_cast<float>(count);
    }

    if (cross.confidence > 0.3f) {
        m.body_height = std::abs(cross.position.z());
    }

    if (bf_top.confidence > 0.3f && bf_bottom.confidence > 0.3f) {
        m.back_fat_thickness = computeDistance(bf_top, bf_bottom);
    }

    if (m.body_width > 0 && m.body_height > 0) {
        m.chest_circumference = 2.0f * (m.body_width * 0.5f + m.body_height * 0.5f) * 3.14159f * 0.5f;
    }

    return m;
}

// ---------------- KeypointSmoother ----------------

KeypointSmoother::KeypointSmoother()
    : alpha_(0.6f), has_history_(false) {}

KeypointSmoother::KeypointSmoother(float alpha)
    : alpha_(std::clamp(alpha, 0.0f, 1.0f)), has_history_(false) {}

void KeypointSmoother::setAlpha(float alpha) {
    alpha_ = std::clamp(alpha, 0.0f, 1.0f);
}

void KeypointSmoother::reset() {
    has_history_ = false;
}

KeypointSet KeypointSmoother::update(const KeypointSet& current) {
    if (!has_history_) {
        history_ = current;
        has_history_ = true;
        return current;
    }

    KeypointSet smoothed;
    for (size_t i = 0; i < current.size(); ++i) {
        smoothed[i].type = current[i].type;
        smoothed[i].confidence = current[i].confidence;
        smoothed[i].position = alpha_ * current[i].position +
                               (1.0f - alpha_) * history_[i].position;
    }
    history_ = smoothed;
    return smoothed;
}

} // namespace swine3d
