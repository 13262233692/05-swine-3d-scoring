#include "common_types.h"
#include "point_cloud_reconstructor.h"
#include "point_cloud_preprocessor.h"
#include "pointnet_inference_engine.h"
#include "keypoint_post_processor.h"
#include "rgb_camera_capture.h"
#include <iostream>
#include <string>
#include <chrono>
#include <cmath>
#include <random>

using namespace swine3d;

void printUsage(const char* prog) {
    std::cout << "Usage:\n"
              << "  " << prog << " demo                    -- Generate synthetic pig & run pipeline\n"
              << "  " << prog << " infer <model.onnx>       -- Load ONNX model + demo data\n"
              << "  " << prog << " camera <model.onnx>      -- Live RealSense capture (if available)\n"
              << "  " << prog << " image <depth> <rgb> <model.onnx> [intrinsics.yaml]\n";
}

PointCloud generateSyntheticPigCloud(int num_points = 5000) {
    PointCloud cloud;
    cloud.reserve(static_cast<size_t>(num_points));
    std::mt19937 rng(12345);
    std::normal_distribution<float> nd(0, 1);
    std::uniform_real_distribution<float> ud(-1, 1);

    for (int i = 0; i < num_points; ++i) {
        float t = ud(rng);
        float body_x = t * 0.6f;
        float body_y = nd(rng) * 0.1f;
        float body_z = 0.7f + 0.15f * std::cos(t * 3.14159f) + nd(rng) * 0.02f;

        uint8_t r = static_cast<uint8_t>(180 + ud(rng) * 40);
        uint8_t g = static_cast<uint8_t>(150 + ud(rng) * 40);
        uint8_t b = static_cast<uint8_t>(120 + ud(rng) * 40);

        cloud.emplace_back(body_x, body_y, body_z, r, g, b);
    }

    for (int i = 0; i < 500; ++i) {
        float gx = ud(rng) * 1.5f;
        float gy = ud(rng) * 1.0f;
        float gz = 0.0f + nd(rng) * 0.02f;
        cloud.emplace_back(gx, gy, gz, 100, 100, 100);
    }

    for (int i = 0; i < 300; ++i) {
        float px = ud(rng) * 1.2f;
        float py = (ud(rng) > 0 ? 0.45f : -0.45f) + nd(rng) * 0.02f;
        float pz = 0.2f + ud(rng) * 0.8f;
        cloud.emplace_back(px, py, pz, 180, 180, 200);
    }

    return cloud;
}

std::vector<Eigen::Vector3f> generateMockKeypoints() {
    std::vector<Eigen::Vector3f> kps;
    kps.emplace_back(-0.55f, -0.08f, 0.78f);
    kps.emplace_back(-0.55f,  0.08f, 0.78f);
    kps.emplace_back(-0.30f, -0.18f, 0.72f);
    kps.emplace_back(-0.30f,  0.18f, 0.72f);
    kps.emplace_back(-0.20f,  0.00f, 0.82f);
    kps.emplace_back( 0.00f, -0.02f, 0.80f);
    kps.emplace_back( 0.00f, -0.02f, 0.75f);
    kps.emplace_back( 0.10f,  0.00f, 0.81f);
    kps.emplace_back( 0.40f, -0.15f, 0.70f);
    kps.emplace_back( 0.40f,  0.15f, 0.70f);
    kps.emplace_back( 0.50f,  0.00f, 0.76f);
    kps.emplace_back( 0.58f,  0.00f, 0.72f);
    return kps;
}

void runFullPipeline(
    PointCloudReconstructor& reconstructor,
    PointCloudPreprocessor& preprocessor,
    PointNetInferenceEngine& engine,
    KeypointPostProcessor& postprocessor,
    KeypointSmoother& smoother,
    const PointCloud& raw_cloud) {

    auto t0 = std::chrono::high_resolution_clock::now();

    PointCloud filtered = preprocessor.applyPassThrough(raw_cloud);
    std::cout << "[Pipeline] Raw points: " << raw_cloud.size()
              << " -> After PassThrough: " << filtered.size() << std::endl;

    PointCloud downsampled = preprocessor.applyVoxelDownsample(filtered);
    std::cout << "[Pipeline] After VoxelDownsample: " << downsampled.size() << std::endl;

    Eigen::Vector3f centroid = PointCloudPreprocessor::computeCentroid(downsampled);
    PointCloud normalized = PointCloudPreprocessor::normalizeCloud(downsampled);

    float scale_factor = 1.0f;
    {
        float max_d2 = 0.0f;
        for (const auto& pt : downsampled) {
            float dx = pt.x - centroid.x();
            float dy = pt.y - centroid.y();
            float dz = pt.z - centroid.z();
            max_d2 = std::max(max_d2, dx*dx + dy*dy + dz*dz);
        }
        scale_factor = std::sqrt(max_d2);
    }

    int N = engine.isLoaded() ? engine.getExpectedPointCount() : 2048;
    PointCloud sampled = preprocessor.randomSampleFixedCount(normalized, N);
    auto eigen_vectors = preprocessor.toEigenVectors(sampled);

    std::vector<Eigen::Vector3f> normalized_keypoints;
    std::vector<float> confidences;

    if (engine.isLoaded()) {
        normalized_keypoints = engine.inferWithConfidence(eigen_vectors, confidences);
        std::cout << "[Pipeline] Inference: " << engine.getLastInferenceTimeMs() << " ms, "
                  << normalized_keypoints.size() << " keypoints" << std::endl;
    } else {
        std::cout << "[Pipeline] No model loaded, using mock keypoints." << std::endl;
        normalized_keypoints = generateMockKeypoints();
        confidences.resize(normalized_keypoints.size(), 0.9f);
    }

    KeypointSet world_kps = postprocessor.mapToWorldSpace(
        normalized_keypoints, confidences, downsampled, centroid, scale_factor);

    KeypointSet smoothed_kps = smoother.update(world_kps);

    BodyMeasurements measurements = KeypointPostProcessor::computeMeasurements(smoothed_kps);

    auto t1 = std::chrono::high_resolution_clock::now();
    float total_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

    std::cout << "\n========== Keypoints ==========" << std::endl;
    for (size_t i = 0; i < smoothed_kps.size(); ++i) {
        const auto& kp = smoothed_kps[i];
        std::cout << "  [" << KeypointPostProcessor::keypointTypeName(kp.type) << "] "
                  << "(" << kp.position.x() << ", "
                  << kp.position.y() << ", "
                  << kp.position.z() << ") m, conf="
                  << kp.confidence << std::endl;
    }

    std::cout << "\n========== Body Measurements ==========" << std::endl;
    std::cout << "  Body length:        " << measurements.body_length * 100.0f << " cm" << std::endl;
    std::cout << "  Body width:         " << measurements.body_width * 100.0f << " cm" << std::endl;
    std::cout << "  Body height:        " << measurements.body_height * 100.0f << " cm" << std::endl;
    std::cout << "  Back-fat thickness: " << measurements.back_fat_thickness * 1000.0f << " mm" << std::endl;
    std::cout << "  Chest circumference:" << measurements.chest_circumference * 100.0f << " cm" << std::endl;

    std::cout << "\nTotal pipeline time: " << total_ms << " ms" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }

    std::string mode = argv[1];

    PointCloudReconstructor reconstructor;
    PointCloudPreprocessor preprocessor;
    PointNetInferenceEngine engine;
    KeypointPostProcessor postprocessor;
    KeypointSmoother smoother(0.7f);

    engine.setIntraOpNumThreads(4);
    engine.setEnableProfiling(true);

    PassThroughParams pt_params;
    pt_params.x_min = -1.0f;
    pt_params.x_max =  1.0f;
    pt_params.y_min = -0.6f;
    pt_params.y_max =  0.6f;
    pt_params.z_min =  0.2f;
    pt_params.z_max =  1.5f;
    preprocessor.setPassThroughParams(pt_params);
    preprocessor.setVoxelGridSize(0.008f);
    preprocessor.setTargetPointCount(2048);

    if (mode == "demo") {
        std::cout << "=== Running DEMO with synthetic pig point cloud ===" << std::endl;
        PointCloud cloud = generateSyntheticPigCloud(8000);
        runFullPipeline(reconstructor, preprocessor, engine,
                        postprocessor, smoother, cloud);

    } else if (mode == "infer" && argc >= 3) {
        std::string model_path = argv[2];
        std::cout << "Loading model: " << model_path << std::endl;
        if (!engine.loadModel(model_path, true, 0)) {
            std::cerr << "Failed to load model, continuing with demo data." << std::endl;
        }
        PointCloud cloud = generateSyntheticPigCloud(8000);
        runFullPipeline(reconstructor, preprocessor, engine,
                        postprocessor, smoother, cloud);

    } else if (mode == "camera" && argc >= 3) {
        std::string model_path = argv[2];
        if (!engine.loadModel(model_path, true, 0)) {
            std::cerr << "Model load failed." << std::endl;
        }

        RGBCameraCapture cam;
        if (!cam.open(0, 640, 480, 30)) {
            std::cerr << "Camera open failed, falling back to demo." << std::endl;
            PointCloud cloud = generateSyntheticPigCloud(8000);
            runFullPipeline(reconstructor, preprocessor, engine,
                            postprocessor, smoother, cloud);
            return 0;
        }

        reconstructor.setIntrinsics(cam.getIntrinsics());
        std::cout << "Capturing... (Ctrl-C to stop)" << std::endl;

        cv::Mat depth, rgb;
        int frame_id = 0;
        while (cam.grabFrames(depth, rgb)) {
            PointCloud cloud = reconstructor.reconstructFromDepthRGB(depth, rgb);
            std::cout << "Frame " << frame_id++ << ": " << cloud.size() << " points" << std::endl;
            runFullPipeline(reconstructor, preprocessor, engine,
                            postprocessor, smoother, cloud);
        }
        cam.close();

    } else if (mode == "image" && argc >= 5) {
        std::string depth_path = argv[2];
        std::string rgb_path   = argv[3];
        std::string model_path = argv[4];
        if (argc >= 6) {
            reconstructor.loadIntrinsicsFromFile(argv[5]);
        }
        if (!engine.loadModel(model_path, true, 0)) {
            std::cerr << "Model load failed." << std::endl;
        }
        cv::Mat depth = PointCloudReconstructor::readDepthImage(depth_path);
        cv::Mat rgb   = PointCloudReconstructor::readRGBImage(rgb_path);
        if (depth.empty()) {
            std::cerr << "Depth image empty: " << depth_path << std::endl;
            return 1;
        }
        PointCloud cloud = reconstructor.reconstructFromDepthRGB(depth, rgb);
        std::cout << "Loaded " << cloud.size() << " points from images." << std::endl;
        runFullPipeline(reconstructor, preprocessor, engine,
                        postprocessor, smoother, cloud);
    } else {
        printUsage(argv[0]);
        return 1;
    }

    return 0;
}
