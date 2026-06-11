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
#include <atomic>
#include <thread>
#include <iomanip>

using namespace swine3d;

void printUsage(const char* prog) {
    std::cout << "Usage:\n"
              << "  " << prog << " demo                    -- Generate synthetic pig & run pipeline\n"
              << "  " << prog << " infer <model.onnx>       -- Load ONNX model + demo data\n"
              << "  " << prog << " batch <model.onnx> [N]   -- 4-pen batch inference (simulate 1-4 pigs)\n"
              << "  " << prog << " stress <model.onnx> [T]  -- Long-running stress test (T seconds)\n"
              << "  " << prog << " camera <model.onnx>      -- Live RealSense capture (if available)\n"
              << "  " << prog << " image <depth> <rgb> <model.onnx> [intrinsics.yaml]\n";
}

PointCloud generateSyntheticPigCloud(int num_points = 5000, int seed = 12345) {
    PointCloud cloud;
    cloud.reserve(static_cast<size_t>(num_points));
    std::mt19937 rng(static_cast<unsigned int>(seed));
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

struct SinglePenResult {
    int pen_id;
    int raw_points;
    int filtered_points;
    Eigen::Vector3f centroid;
    float scale_factor;
    KeypointSet keypoints;
    BodyMeasurements measurements;
    float inference_ms;
    float gpu_mem_mb;
};

std::vector<SinglePenResult> runBatchPipeline(
    PointCloudReconstructor& reconstructor,
    PointCloudPreprocessor& preprocessor,
    PointNetInferenceEngine& engine,
    KeypointPostProcessor& postprocessor,
    std::vector<KeypointSmoother>& smoothers,
    const std::vector<PointCloud>& raw_clouds,
    bool use_mock_fallback = true) {

    auto t0 = std::chrono::high_resolution_clock::now();
    const int num_pens = static_cast<int>(raw_clouds.size());

    std::vector<PointCloud> filtered_clouds(num_pens);
    std::vector<PointCloud> downsampled_clouds(num_pens);
    std::vector<Eigen::Vector3f> centroids(num_pens);
    std::vector<float> scale_factors(num_pens);
    std::vector<std::vector<Eigen::Vector3f>> eigen_batches(num_pens);

    int N = engine.isLoaded() ? engine.getExpectedPointCount() : 2048;

    for (int p = 0; p < num_pens; ++p) {
        filtered_clouds[p] = preprocessor.applyPassThrough(raw_clouds[p]);
        downsampled_clouds[p] = preprocessor.applyVoxelDownsample(filtered_clouds[p]);
        centroids[p] = PointCloudPreprocessor::computeCentroid(downsampled_clouds[p]);

        float max_d2 = 0.0f;
        for (const auto& pt : downsampled_clouds[p]) {
            float dx = pt.x - centroids[p].x();
            float dy = pt.y - centroids[p].y();
            float dz = pt.z - centroids[p].z();
            max_d2 = std::max(max_d2, dx*dx + dy*dy + dz*dz);
        }
        scale_factors[p] = std::sqrt(std::max(1e-6f, max_d2));

        PointCloud normalized = PointCloudPreprocessor::normalizeCloud(downsampled_clouds[p]);
        PointCloud sampled = preprocessor.randomSampleFixedCount(normalized, N);
        eigen_batches[p] = preprocessor.toEigenVectors(sampled);
    }

    PointNetInferenceEngine::BatchKeypoints batch_keypoints;
    PointNetInferenceEngine::BatchConfidences batch_confidences;
    float inference_ms = 0.0f;

    if (engine.isLoaded()) {
        batch_keypoints = engine.inferBatchWithConfidence(eigen_batches, batch_confidences);
        inference_ms = engine.getLastInferenceTimeMs();
    } else if (use_mock_fallback) {
        std::cout << "[Batch] No model loaded, using mock keypoints." << std::endl;
        batch_keypoints.resize(static_cast<size_t>(num_pens));
        batch_confidences.resize(static_cast<size_t>(num_pens));
        auto mock = generateMockKeypoints();
        for (int p = 0; p < num_pens; ++p) {
            batch_keypoints[p] = mock;
            batch_confidences[p].assign(mock.size(), 0.9f);
        }
    }

    float gpu_mem = engine.getCurrentGpuMemoryUsedMb();

    std::vector<SinglePenResult> results(num_pens);
    for (int p = 0; p < num_pens; ++p) {
        SinglePenResult& r = results[p];
        r.pen_id = p;
        r.raw_points = static_cast<int>(raw_clouds[p].size());
        r.filtered_points = static_cast<int>(downsampled_clouds[p].size());
        r.centroid = centroids[p];
        r.scale_factor = scale_factors[p];
        r.inference_ms = inference_ms;
        r.gpu_mem_mb = gpu_mem;

        if (p < static_cast<int>(batch_keypoints.size())) {
            r.keypoints = postprocessor.mapToWorldSpace(
                batch_keypoints[p], batch_confidences[p],
                downsampled_clouds[p], centroids[p], scale_factors[p]);

            if (p < static_cast<int>(smoothers.size())) {
                r.keypoints = smoothers[p].update(r.keypoints);
            }

            r.measurements = KeypointPostProcessor::computeMeasurements(r.keypoints);
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    float total_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

    std::cout << "\n========== Batch Summary (batch=" << num_pens << ") ==========" << std::endl;
    std::cout << "  Total pipeline: " << total_ms << " ms"
              << "  |  Inference: " << inference_ms << " ms"
              << "  |  GPU: " << gpu_mem << " MB" << std::endl;

    for (int p = 0; p < num_pens; ++p) {
        const auto& r = results[p];
        std::cout << "\n--- Pen " << p << " ---" << std::endl;
        std::cout << "  Points: " << r.raw_points << " -> " << r.filtered_points << std::endl;
        std::cout << "  Body length: " << r.measurements.body_length * 100.0f << " cm" << std::endl;
        std::cout << "  Body width:  " << r.measurements.body_width * 100.0f << " cm" << std::endl;
        std::cout << "  Back-fat:    " << r.measurements.back_fat_thickness * 1000.0f << " mm" << std::endl;
    }

    return results;
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
    std::vector<KeypointSmoother> smoothers(4, KeypointSmoother(0.7f));

    engine.setIntraOpNumThreads(4);
    engine.setEnableProfiling(true);
    engine.setOomThresholdMb(3500.0f);

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

    auto runSingleDemo = [&](const std::string& model_path = "") {
        if (!model_path.empty()) {
            std::cout << "Loading model: " << model_path << std::endl;
            if (!engine.loadModel(model_path, true, 0, PointNetInferenceEngine::MAX_BATCH_SIZE)) {
                std::cerr << "Failed to load model, continuing with demo data." << std::endl;
            }
        }
        PointCloud cloud = generateSyntheticPigCloud(8000);
        std::vector<PointCloud> batch_clouds(1, cloud);
        runBatchPipeline(reconstructor, preprocessor, engine,
                        postprocessor, smoothers, batch_clouds, true);
    };

    if (mode == "demo") {
        std::cout << "=== Running DEMO with synthetic pig point cloud ===" << std::endl;
        runSingleDemo();

    } else if (mode == "infer" && argc >= 3) {
        runSingleDemo(argv[2]);

    } else if (mode == "batch" && argc >= 3) {
        std::string model_path = argv[2];
        int num_pigs = 4;
        if (argc >= 4) num_pigs = std::max(1, std::min(4, std::atoi(argv[3])));

        std::cout << "Loading model: " << model_path << " for batch=" << num_pigs << std::endl;
        if (!engine.loadModel(model_path, true, 0, PointNetInferenceEngine::MAX_BATCH_SIZE)) {
            std::cerr << "Model load failed, using mock fallback." << std::endl;
        }

        std::cout << "\n=== Batch Inference: " << num_pigs << " pigs from 4 pens ===" << std::endl;
        std::vector<PointCloud> batch_clouds;
        batch_clouds.reserve(static_cast<size_t>(num_pigs));
        for (int i = 0; i < num_pigs; ++i) {
            batch_clouds.push_back(generateSyntheticPigCloud(8000, 10000 + i * 7));
        }

        runBatchPipeline(reconstructor, preprocessor, engine,
                        postprocessor, smoothers, batch_clouds);

    } else if (mode == "stress" && argc >= 3) {
        std::string model_path = argv[2];
        int duration_sec = 300;
        if (argc >= 4) duration_sec = std::max(10, std::atoi(argv[3]));

        std::cout << "Loading model for stress test: " << model_path << std::endl;
        if (!engine.loadModel(model_path, true, 0, PointNetInferenceEngine::MAX_BATCH_SIZE)) {
            std::cerr << "Model load failed!" << std::endl;
            return 1;
        }

        std::cout << "\n=== Stress Test: " << duration_sec << " seconds, "
                  << "4 pens with dynamic batch=1~4 ===" << std::endl;

        std::mt19937 rng(98765);
        std::uniform_int_distribution<int> batch_dist(1, 4);

        auto start_time = std::chrono::steady_clock::now();
        int iteration = 0;
        int oom_events = 0;
        float max_gpu_mb = 0.0f;
        float min_gpu_mb = std::numeric_limits<float>::max();
        double total_inference_ms = 0.0;
        int total_pens_processed = 0;

        while (true) {
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - start_time).count();
            if (elapsed >= duration_sec) break;

            int dynamic_batch = batch_dist(rng);
            std::vector<PointCloud> batch_clouds;
            batch_clouds.reserve(static_cast<size_t>(dynamic_batch));
            for (int i = 0; i < dynamic_batch; ++i) {
                batch_clouds.push_back(generateSyntheticPigCloud(8000, iteration * 100 + i));
            }

            auto results = runBatchPipeline(reconstructor, preprocessor, engine,
                                           postprocessor, smoothers, batch_clouds);

            float gpu_mb = engine.getCurrentGpuMemoryUsedMb();
            max_gpu_mb = std::max(max_gpu_mb, gpu_mb);
            min_gpu_mb = std::min(min_gpu_mb, gpu_mb);
            total_inference_ms += engine.getLastInferenceTimeMs();
            total_pens_processed += dynamic_batch;

            if (iteration % 50 == 0) {
                std::cout << "\n[STRESS] Iteration " << iteration
                          << ", Elapsed=" << elapsed << "s / " << duration_sec << "s"
                          << ", GPU min=" << min_gpu_mb << "MB max=" << max_gpu_mb << "MB"
                          << ", Avg inf=" << (total_inference_ms / (iteration + 1)) << "ms"
                          << ", OOM events=" << oom_events
                          << std::endl;
            }

            if (gpu_mb > engine.getCurrentGpuMemoryUsedMb() * 0.98f) {
                ++oom_events;
            }

            ++iteration;
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }

        std::cout << "\n========== STRESS TEST SUMMARY ==========" << std::endl;
        std::cout << "  Duration:          " << duration_sec << " seconds" << std::endl;
        std::cout << "  Total iterations:  " << iteration << std::endl;
        std::cout << "  Total pens:        " << total_pens_processed << std::endl;
        std::cout << "  Avg GPU memory:    " << (min_gpu_mb + max_gpu_mb) / 2.0f << " MB" << std::endl;
        std::cout << "  Peak GPU memory:   " << max_gpu_mb << " MB" << std::endl;
        std::cout << "  Trough GPU memory: " << min_gpu_mb << " MB" << std::endl;
        std::cout << "  GPU memory delta:  " << (max_gpu_mb - min_gpu_mb) << " MB (fragmentation indicator)" << std::endl;
        std::cout << "  Avg inference:     " << (total_inference_ms / iteration) << " ms / batch" << std::endl;
        std::cout << "  OOM events:        " << oom_events << std::endl;

        if ((max_gpu_mb - min_gpu_mb) > 100.0f) {
            std::cout << "\n  [WARNING] Significant memory drift detected (>100MB)." << std::endl;
        } else {
            std::cout << "\n  [OK] Memory footprint stable. Static pool is working." << std::endl;
        }

    } else if (mode == "camera" && argc >= 3) {
        std::string model_path = argv[2];
        if (!engine.loadModel(model_path, true, 0, PointNetInferenceEngine::MAX_BATCH_SIZE)) {
            std::cerr << "Model load failed." << std::endl;
        }

        RGBCameraCapture cam;
        if (!cam.open(0, 640, 480, 30)) {
            std::cerr << "Camera open failed, falling back to demo." << std::endl;
            runSingleDemo();
            return 0;
        }

        reconstructor.setIntrinsics(cam.getIntrinsics());
        std::cout << "Capturing... (Ctrl-C to stop)" << std::endl;

        cv::Mat depth, rgb;
        int frame_id = 0;
        while (cam.grabFrames(depth, rgb)) {
            PointCloud cloud = reconstructor.reconstructFromDepthRGB(depth, rgb);
            std::cout << "Frame " << frame_id++ << ": " << cloud.size() << " points" << std::endl;
            std::vector<PointCloud> batch_clouds(1, cloud);
            runBatchPipeline(reconstructor, preprocessor, engine,
                            postprocessor, smoothers, batch_clouds);
        }
        cam.close();

    } else if (mode == "image" && argc >= 5) {
        std::string depth_path = argv[2];
        std::string rgb_path   = argv[3];
        std::string model_path = argv[4];
        if (argc >= 6) {
            reconstructor.loadIntrinsicsFromFile(argv[5]);
        }
        if (!engine.loadModel(model_path, true, 0, PointNetInferenceEngine::MAX_BATCH_SIZE)) {
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
        std::vector<PointCloud> batch_clouds(1, cloud);
        runBatchPipeline(reconstructor, preprocessor, engine,
                        postprocessor, smoothers, batch_clouds);
    } else {
        printUsage(argv[0]);
        return 1;
    }

    engine.shutdown();
    return 0;
}
