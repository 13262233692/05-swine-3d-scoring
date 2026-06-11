#include "rgb_camera_capture.h"
#include <iostream>

namespace swine3d {

RGBCameraCapture::RGBCameraCapture()
#ifdef USE_REALSENSE
    : align_to_color_(true)
#endif
    , opened_(false)
    , width_(640)
    , height_(480) {}

RGBCameraCapture::~RGBCameraCapture() {
    close();
}

bool RGBCameraCapture::open(int device_index, int width, int height, int fps) {
#ifdef USE_REALSENSE
    try {
        width_ = width;
        height_ = height;

        config_.enable_stream(RS2_STREAM_DEPTH, width, height, RS2_FORMAT_Z16, fps);
        config_.enable_stream(RS2_STREAM_COLOR, width, height, RS2_FORMAT_BGR8, fps);

        auto profile = pipeline_.start(config_);

        auto depth_stream = profile.get_stream(RS2_STREAM_DEPTH)
                               .as<rs2::video_stream_profile>();
        depth_intrinsics_ = depth_stream.get_intrinsics();

        auto color_stream = profile.get_stream(RS2_STREAM_COLOR)
                               .as<rs2::video_stream_profile>();
        color_intrinsics_ = color_stream.get_intrinsics();

        opened_ = true;
        return true;
    } catch (const rs2::error& e) {
        std::cerr << "RealSense error: " << e.what() << std::endl;
        opened_ = false;
        return false;
    }
#else
    (void)device_index;
    (void)width;
    (void)height;
    (void)fps;
    std::cerr << "RealSense support not compiled in." << std::endl;
    return false;
#endif
}

bool RGBCameraCapture::isOpened() const {
    return opened_;
}

void RGBCameraCapture::close() {
#ifdef USE_REALSENSE
    if (opened_) {
        try {
            pipeline_.stop();
        } catch (...) {}
        opened_ = false;
    }
#endif
}

bool RGBCameraCapture::grabFrames(cv::Mat& out_depth, cv::Mat& out_rgb) {
#ifdef USE_REALSENSE
    if (!opened_) return false;
    try {
        frames_ = pipeline_.wait_for_frames(5000);

        rs2::frame depth_frame;
        rs2::frame color_frame;

        if (align_to_color_) {
            rs2::align align(RS2_STREAM_COLOR);
            auto aligned = align.process(frames_);
            depth_frame = aligned.get_depth_frame();
            color_frame = aligned.get_color_frame();
        } else {
            depth_frame = frames_.get_depth_frame();
            color_frame = frames_.get_color_frame();
        }

        const int depth_w = depth_frame.as<rs2::video_frame>().get_width();
        const int depth_h = depth_frame.as<rs2::video_frame>().get_height();
        const int color_w = color_frame.as<rs2::video_frame>().get_width();
        const int color_h = color_frame.as<rs2::video_frame>().get_height();

        out_depth = cv::Mat(depth_h, depth_w, CV_16UC1,
                            (void*)depth_frame.get_data(), cv::Mat::AUTO_STEP).clone();
        out_rgb = cv::Mat(color_h, color_w, CV_8UC3,
                          (void*)color_frame.get_data(), cv::Mat::AUTO_STEP).clone();
        return true;
    } catch (const rs2::error& e) {
        std::cerr << "Frame grab error: " << e.what() << std::endl;
        return false;
    }
#else
    (void)out_depth;
    (void)out_rgb;
    return false;
#endif
}

CameraIntrinsics RGBCameraCapture::getIntrinsics() const {
    CameraIntrinsics ci;
#ifdef USE_REALSENSE
    if (opened_) {
        const auto& intr = align_to_color_ ? color_intrinsics_ : depth_intrinsics_;
        ci.fx = intr.fx;
        ci.fy = intr.fy;
        ci.cx = intr.ppx;
        ci.cy = intr.ppy;
        ci.width = intr.width;
        ci.height = intr.height;
        ci.depth_scale = 0.001f;
    }
#endif
    return ci;
}

bool RGBCameraCapture::saveSnapshot(const std::string& depth_path, const std::string& rgb_path) {
    cv::Mat depth, rgb;
    if (!grabFrames(depth, rgb)) return false;
    cv::imwrite(depth_path, depth);
    cv::imwrite(rgb_path, rgb);
    return true;
}

void RGBCameraCapture::setAlignToColor(bool align) {
#ifdef USE_REALSENSE
    align_to_color_ = align;
#else
    (void)align;
#endif
}

} // namespace swine3d
