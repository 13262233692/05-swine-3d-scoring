# 05-swine-3d-scoring

智慧农业 AI 边缘计算项目 —— 基于 RGB-D 深度相机与 PointNet++ 的生猪三维体测关键点提取引擎。

## 系统架构

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                   猪舍顶部边缘盒子 (Jetson / x86_64 + CUDA)                   │
│                                                                              │
│  ┌──────────────┐   ┌────────────────────┐   ┌───────────────────┐           │
│  │ Intel D435i  │──▶│  16-bit 深度解码    │──▶│ 反投影(内参矩阵)  │           │
│  │×4 栏RGB-D相机│   │  + RGB 对齐 (60fps) │   │ → 稠密彩色点云    │           │
│  └──────────────┘   └────────────────────┘   └────────┬──────────┘           │
│                                                        │                      │
│                              ┌─────────────────────────▼──────────┐           │
│                              │  点云预处理 (PassThrough + Voxel)   │           │
│                              │  - 切除水泥地面/钢管背景            │           │
│                              │  - 体素下采样 → 固定 2048 点        │           │
│                              └─────────────────────────┬──────────┘           │
│                                                        │                      │
│               ┌────────────────────────────────────────┘                      │
│               ▼                                                                │
│   ┌──────────────────────────────────────────────────────────────┐             │
│   │   静态CUDA显存池 + 批量推理 (Batch=1~4)                      │             │
│   │   PointNet++ (ONNX Runtime + TensorRT + CUDA EP)            │             │
│   │   输入: [1~4, 2048, 3]  →  输出: [1~4, 12, 4] (xyz+conf)   │             │
│   └─────────────────────────┬────────────────────────────────────┘             │
│                             │                                                  │
│            ┌────────────────▼──────────────────┐                               │
│            │   脊背6锚点 B-Spline 曲面拟合     │                               │
│            │   最小二乘 + de Boor-Cox 基函数    │                               │
│            │   → 光滑猪背部物理轮廓曲面         │                               │
│            └────────────────┬──────────────────┘                               │
│                             │                                                  │
│            ┌────────────────▼──────────────────┐                               │
│            │  48切片 Simpson 体积微积分         │                               │
│            │  腹底距ground_z积分 + 背膘体积     │                               │
│            │  → 总体积 / 腹腔容 / 背膘脂 L      │                               │
│            └────────────────┬──────────────────┘                               │
│                             │                                                  │
│            ┌────────────────▼──────────────────┐                               │
│            │  BCS体况评分 (1~5分制)             │                               │
│            │  脂肪/肌肉/骨架三维特征加权融合     │                               │
│            └────────────────┬──────────────────┘                               │
│                             │                                                  │
│            ┌────────────────▼──────────────────┐                               │
│            │  异速生长 估重                     │                               │
│            │  W = α·Volume·ρ·BCS + a·L^b       │                               │
│            │  → 精准到0.1kg的体重报表           │                               │
│            └───────────────────────────────────┘                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

## 项目结构

```
05-swine-3d-scoring/
├── CMakeLists.txt
├── include/
│   ├── common_types.h                    # 数据类型定义 (点云、关键点、体测…)
│   ├── point_cloud_reconstructor.h       # 深度图 → 3D 点云反投影
│   ├── point_cloud_preprocessor.h        # PassThrough / Voxel / 采样 / 归一化
│   ├── pointnet_inference_engine.h       # ONNX Runtime + TensorRT 推理封装
│   ├── keypoint_post_processor.h         # 反归一化/KNN投影/EMA平滑/体测
│   └── rgb_camera_capture.h              # Intel RealSense 采集封装
├── src/
│   ├── main.cpp                          # 主程序 (demo / infer / camera / image)
│   ├── point_cloud_reconstructor.cpp
│   ├── point_cloud_preprocessor.cpp
│   ├── pointnet_inference_engine.cpp
│   ├── keypoint_post_processor.cpp
│   └── rgb_camera_capture.cpp
└── config/
    ├── camera_intrinsics.yaml            # 相机内参 (fx, fy, cx, cy, depth_scale)
    └── pipeline_config.yaml              # 直通滤波、体素大小、推理参数
```

## 12 个核心关键点 (KeypointType)

| 索引 | 枚举 | 说明 |
|------|------|------|
| 0 | LEFT_EAR | 左耳尖 |
| 1 | RIGHT_EAR | 右耳尖 |
| 2 | LEFT_SHOULDER | 左肩端 |
| 3 | RIGHT_SHOULDER | 右肩端 |
| 4 | WITHERS | 鬐甲 (体高点) |
| 5 | BACK_FAT_TOP | 背膘检测点(上) |
| 6 | BACK_FAT_BOTTOM | 背膘检测点(下) |
| 7 | CROSS_SECTION | 十字部 |
| 8 | LEFT_HIP | 左髋关节 |
| 9 | RIGHT_HIP | 右髋关节 |
| 10 | RUMP | 臀端 |
| 11 | TAIL_HEAD | 尾根 |

## 构建依赖

- **C++17** 编译器 (GCC 9+, Clang 10+, MSVC 2019+)
- **CMake ≥ 3.18**
- **OpenCV ≥ 4.5** — 图像读写与色彩转换
- **Eigen3 ≥ 3.3** — 线性代数/向量运算
- **ONNX Runtime ≥ 1.14** (放入 `third_party/onnxruntime/`)
  - 头文件: `third_party/onnxruntime/include/`
  - 库文件: `third_party/onnxruntime/lib/`
- **TensorRT ≥ 8.5** (可选, 环境变量 `TENSORRT_ROOT`)
- **CUDA ≥ 11.6** (可选, TensorRT 依赖)
- **Intel RealSense SDK ≥ 2.50** (可选, 实时采集)

### Windows 快速构建

```powershell
# 假设 vcpkg 已安装 opencv4 eigen3 realsense2
cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake `
      -DONNXRUNTIME_ROOT="D:/libs/onnxruntime-win-x64-gpu-1.16.3" `
      -DTENSORRT_ROOT="D:/TensorRT-8.6.1.6"
cmake --build build --config Release
```

## 运行模式

```bash
# 1. 纯演示 (合成猪只点云 + mock 关键点, 无需任何硬件/模型)
./swine_3d_scoring demo

# 2. 加载模型 + 合成数据推理
./swine_3d_scoring infer models/pointnet2_keypoints.onnx

# 3. 实时相机采集 (RealSense + 模型)
./swine_3d_scoring camera models/pointnet2_keypoints.onnx

# 4. 离线图像推理 (已保存的 depth.png + rgb.png)
./swine_3d_scoring image data/frame_0001_depth.png data/frame_0001_rgb.png \
    models/pointnet2_keypoints.onnx config/camera_intrinsics.yaml
```

## 核心技术要点

### 1. 深度反投影 (Back-projection)

给定像素 `(u, v)` 和深度值 `Z_m = depth_raw * depth_scale`:

```
X = (u - cx) * Z_m / fx
Y = (v - cy) * Z_m / fy
Z = Z_m
```

参见 [point_cloud_reconstructor.cpp#L61-L66](file:///d:/SOLO-11/05-swine-3d-scoring/src/point_cloud_reconstructor.cpp#L61-L66)

### 2. 直通滤波 (PassThrough)

切除猪栏外水泥地面 (z < 0.2m) 、两侧钢管 (|y| > 0.6m) 等干扰背景。

参见 [point_cloud_preprocessor.cpp#L57-L69](file:///d:/SOLO-11/05-swine-3d-scoring/src/point_cloud_preprocessor.cpp#L57-L69)

### 3. PointNet++ ONNX 推理

模型期望输入形状 `[batch=1, N=2048, C=3]` (float32), 输出 `[1, 12, 3]` 或 `[1, 12, 4]` (含置信度)。
可使用 `AppendExecutionProvider_TensorRT` 启用 FP16 加速。

参见 [pointnet_inference_engine.cpp#L30-L65](file:///d:/SOLO-11/05-swine-3d-scoring/src/pointnet_inference_engine.cpp#L30-L65)

### 4. 反归一化 + KNN 投影

网络输出是归一化空间 (centered & unit-farthest-distance) 的坐标, 需:
1. 乘以下采样时保存的 `scale_factor`, 加上 `centroid` 回到物理空间
2. 取点云中 K=3 最近点的平均, 将关键点 "贴" 回猪体表面, 避免浮于体外

参见 [keypoint_post_processor.cpp#L136-L157](file:///d:/SOLO-11/05-swine-3d-scoring/src/keypoint_post_processor.cpp#L136-L157)

### 5. EMA 时序平滑

猪只一直在扭动, 对相邻帧的关键点做指数移动平均 (α=0.7):
```
P_smooth[t] = α * P_raw[t] + (1 - α) * P_smooth[t-1]
```

参见 [keypoint_post_processor.cpp#L234-L251](file:///d:/SOLO-11/05-swine-3d-scoring/src/keypoint_post_processor.cpp#L234-L251)

## 🔴 显存 OOM 问题深度修复 (CUDA 静态显存池)

### 问题根因分析

原始实现在 4 栏并发 + 动态 batch=1~4 场景下运行 10 分钟即 OOM，核心原因：

| 缺陷 | 现象 |
|------|------|
| 每次推理 `std::vector<float> input_data(N*3)` | CPU 侧频繁分配释放 |
| `Ort::Value::CreateTensor` + 动态 batch | ORT/TRT 内部每次调用 `cudaMalloc` |
| 无显式 CUDA 同步 | `cudaFree` 异步延迟导致碎片化 |
| 无预分配机制 | 小显存块被不断分配释放形成空洞 |

### 修复方案：静态极大显存池 + 手动张量偏移

```
CUDA Memory Layout (max_batch=4, N=2048, K=12, CD=4)
┌───────────────────────────────────────────────────────────┐
│ d_input (96 KB total)                                     │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐      │
│  │ pen0[0] │  │ pen1[0] │  │ pen2[0] │  │ pen3[0] │      │
│  │  2048×3 │  │  2048×3 │  │  2048×3 │  │  2048×3 │      │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘      │
│   stride = 24 KB                                            │
├───────────────────────────────────────────────────────────┤
│ d_output (0.75 KB total)                                  │
│  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐         │
│  │ pen0[0]│  │ pen1[0]│  │ pen2[0]│  │ pen3[0]│         │
│  │  12×4  │  │  12×4  │  │  12×4  │  │  12×4  │         │
│  └────────┘  └────────┘  └────────┘  └────────┘         │
└───────────────────────────────────────────────────────────┘

Input offset for pen b:  d_input + b * N * 3
Output offset for pen b: d_output + b * K * coord_dim
```

**关键实现位置：**

1. **静态池预分配** (模型加载时一次性完成，绝不在循环内)
   - [allocateStaticMemoryPool()](file:///d:/SOLO-11/05-swine-3d-scoring/src/pointnet_inference_engine.cpp#L126-L192) — `cudaMalloc` + `cudaHostAlloc` (pinned)
   - 输入池: `4 × 2048 × 3 × 4B = 96 KB`
   - 输出池: `4 × 12 × 4 × 4B = 0.75 KB`
   - 锁页 CPU 镜像池: ×2 用于异步 H2D/D2H

2. **手动偏移步长** (消除动态 `cudaMalloc`)
   - [copyBatchToDevice()](file:///d:/SOLO-11/05-swine-3d-scoring/src/pointnet_inference_engine.cpp#L414-L444) — H2D 按 `stride = N × 3 × sizeof(float)` 偏移
   - [copyBatchFromDevice()](file:///d:/SOLO-11/05-swine-3d-scoring/src/pointnet_inference_engine.cpp#L446-L495) — D2H 按 `stride = K × coord_dim × sizeof(float)` 偏移
   - [inferBatchWithConfidence()](file:///d:/SOLO-11/05-swine-3d-scoring/src/pointnet_inference_engine.cpp#L526-L595) — Ort::Value 直接绑定预分配指针 + 动态 shape

3. **显式生命周期管理**
   - [shutdown()](file:///d:/SOLO-11/05-swine-3d-scoring/src/pointnet_inference_engine.cpp#L98-L116) — 锁 → 同步 → 释放池 → 销毁流 → 析构 session
   - [synchronizeCuda()](file:///d:/SOLO-11/05-swine-3d-scoring/src/pointnet_inference_engine.cpp#L118-L124) — 非阻塞流 `cudaStreamSynchronize`

4. **OOM 熔断机制**
   - [checkOomProtection()](file:///d:/SOLO-11/05-swine-3d-scoring/src/pointnet_inference_engine.cpp#L246-L265) — 推理前 `cudaMemGetInfo` 检查，超阈值强制 purge
   - `setOomThresholdMb(3500)` — 可配置阈值，默认 3.5 GB

### 压力测试验证

```bash
# 运行 300 秒动态 batch=1~4 压力测试
./swine_3d_scoring stress models/pointnet2_keypoints.onnx 300
```

预期输出 (显存稳定指标):
```
========== STRESS TEST SUMMARY ==========
  Duration:          300 seconds
  Total iterations:  ~12000
  Total pens:        ~30000
  Peak GPU memory:   1285 MB
  Trough GPU memory: 1278 MB
  GPU memory delta:  7 MB (fragmentation indicator)
  [OK] Memory footprint stable. Static pool is working.
```

**验证标准**: `max_gpu_mb - min_gpu_mb < 100 MB` 表示无泄漏。

## PointNet++ 模型导出 (PyTorch → ONNX)

```python
import torch
import onnx
from pointnet2_ops import pointnet2_utils

# 假设已训练的模型输出 [B, 12, 3] 关键点坐标
model = YourPointNet2KeypointNet(num_keypoints=12)
model.load_state_dict(torch.load("pointnet2_keypoints.pth"))
model.eval()

dummy = torch.randn(1, 2048, 3)
torch.onnx.export(
    model, dummy, "pointnet2_keypoints.onnx",
    input_names=["points"],
    output_names=["keypoints"],
    opset_version=17,
    dynamic_axes={"points": {0: "batch"}, "keypoints": {0: "batch"}},
)
```
