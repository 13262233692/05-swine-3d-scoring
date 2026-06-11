#pragma once

#include "common_types.h"
#include <string>

#ifdef USE_REALSENSE
#include <librealsense2/rs.hpp>
#endif

namespace swine3d {

class RGBCameraCapture {
public:
    RGBCameraCapture();
    ~RGBCameraCapture();

    bool open(int device_index = 0, int width = 640, int height = 480, int fps = 30);

    bool isOpened() const;

    void close();

    bool grabFrames(cv::Mat& out_depth, cv::Mat& out_rgb);

    CameraIntrinsics getIntrinsics() const;

    bool saveSnapshot(const std::string& depth_path, const std::string& rgb_path);

    void setAlignToColor(bool align);

private:
#ifdef USE_REALSENSE
    rs2::pipeline pipeline_;
    rs2::config config_;
    rs2::frameset frames_;
    rs2_intrinsics depth_intrinsics_;
    rs2_intrinsics color_intrinsics_;
    bool align_to_color_;
#endif
    bool opened_;
    int width_;
    int height_;
};

} // namespace swine3d
