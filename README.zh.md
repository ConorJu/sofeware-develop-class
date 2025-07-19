# 🚗 YOLO 交通计数器

一个使用 YOLO 目标检测的综合交通分析系统，用于计数车辆和行人，具有实时可视化和高级跟踪能力。

## ✨ 特性

### 🎯 核心功能
- **实时交通计数**：计数跨越指定线的车辆和行人
- **对象跟踪**：高级多对象跟踪与轨迹分析
- **视频处理**：处理各种视频格式 (MP4, AVI, MOV, MKV)
- **可定制检测**：可调整置信阈值和计数线

### 🎓 模型训练
- **自定义模型训练**：在您自己的数据集上训练 YOLO 模型
- **多种模型尺寸**：支持 YOLOv8n/s/m/l/x 变体
- **自动化数据集准备**：轻松的数据集分割和组织
- **训练进度监控**：实时训练指标和可视化

### 📝 数据管理
- **图像标注工具**：交互式 GUI 用于标记对象
- **格式转换**：在 XML (Pascal VOC) 和 TXT (YOLO) 格式之间转换
- **数据验证**：标注文件的自动验证
- **批量处理**：同时处理多个文件

### 🌐 Web 界面
- **现代 UI**：基于 Gradio 的美丽 web 界面
- **实时分析**：交互式图表和可视化
- **进度跟踪**：实时处理状态和统计
- **多标签界面**：不同任务的组织化工作流

## 🚀 快速开始

### 安装

1. **克隆仓库**
```bash
git clone https://github.com/your-repo/yolo-traffic-counter.git
cd yolo-traffic-counter
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

3. **启动 web 界面**
```bash
python main.py web
```

导航到 `http://localhost:8501` 访问 web 界面。

## 📋 要求

- Python 3.8+
- 支持 CUDA 的 GPU（推荐用于训练和推理）
- 8GB+ RAM
- 支持 OpenCV 的系统

### 依赖
- PyTorch >= 1.9.0
- Ultralytics YOLO >= 8.0.0
- OpenCV >= 4.5.0
- Gradio >= 3.0.0
- Plotly >= 5.0.0
- 以及更多（见 requirements.txt）

## 🎯 使用

### 命令行界面

项目提供统一的 CLI 用于所有操作：

#### 1. 启动 Web 界面
```bash
# 基本启动
python main.py web

# 自定义主机和端口
python main.py web --host 0.0.0.0 --port 8080

# 公共分享（创建公共 URL）
python main.py web --share
```

#### 2. 在视频中计数交通
```bash
# 基本计数
python main.py count --input video.mp4 --output result.mp4

# 自定义设置
python main.py count --input video.mp4 --output result.mp4 --conf 0.6 --line-y 400

# 处理期间显示视频
python main.py count --input video.mp4 --show
```

#### 3. 训练自定义模型
```bash
# 使用数据集训练
python main.py train --data dataset.yaml --epochs 100 --batch 16

# 使用更大的模型
python main.py train --data dataset.yaml --model yolov8m --epochs 200
```

#### 4. 标注图像
```bash
# 标注单个图像
python main.py annotate --input image.jpg

# 标注图像目录
python main.py annotate --input images/
```

#### 5. 转换标注格式
```bash
# XML 到 YOLO 格式
python main.py convert --mode xml2yolo --input annotations/ --output yolo_labels/

# YOLO 到 XML 格式
python main.py convert --mode yolo2xml --input yolo_labels/ --images images/ --output xml_annotations/

# 验证标注
python main.py convert --mode validate --input annotations/ --output validation_report/
```

#### 6. 运行目标检测
```bash
# 在图像中检测对象
python main.py detect --input image.jpg --output result.jpg --show

# 处理视频
python main.py detect --input video.mp4 --output detected.mp4

# 基准性能
python main.py detect --input test.jpg --benchmark
```

#### 7. 准备数据集
```bash
python main.py prepare --images images/ --annotations annotations/ --output data/dataset --split 0.8 0.2
```

### Web 界面使用

1. **交通计数标签**：
   - 上传视频文件
   - 调整置信阈值和计数线位置
   - 点击“处理视频”开始计数
   - 查看带有交互式图表的结果

2. **模型训练标签**：
   - 指定图像和标注文件夹
   - 配置训练参数
   - 开始训练过程
   - 监控进度

3. **数据管理标签**：
   - 在标注格式之间转换
   - 验证标注文件
   - 批量处理多个文件

4. **模型管理标签**：
   - 使用自定义权重初始化模型
   - 查看模型信息和统计

## 📊 配置

系统使用位于 `configs/config.yaml` 的 YAML 配置文件。主要设置包括：

```yaml
# 模型配置
model:
  name: "yolov8n"
  num_classes: 2
  input_size: 640

# 训练配置  
training:
  epochs: 100
  batch_size: 16
  learning_rate: 0.001

# 检测配置
detection:
  confidence_threshold: 0.5
  nms_threshold: 0.4

# 计数配置
counting:
  line_position: 0.5
  track_history: 30
  min_track_length: 5
```

## 🏗️ 项目结构

```
yolo-traffic-counter/
├── configs/                 # 配置文件
│   └── config.yaml
├── data/                    # 数据目录
│   ├── raw/                # 原始输入数据
│   ├── processed/          # 处理后的数据
│   ├── annotations/        # 标注文件
│   └── dataset/           # 准备好的数据集
├── models/                 # 模型权重
├── src/                   # 源代码
│   ├── annotation/        # 数据标注工具
│   ├── detection/         # 目标检测和计数
│   ├── training/          # 模型训练
│   └── frontend/          # Web 界面
├── utils/                 # 实用函数
├── tests/                 # 测试文件
├── runs/                  # 训练运行和结果
├── logs/                  # 日志文件
├── main.py               # 主入口点
├── requirements.txt      # Python 依赖
└── README.md            # 本文件
```

## 🔧 高级使用

### 自定义模型训练

1. **准备数据集**：
   ```bash
   # 组织图像和标注
   mkdir -p data/custom/{images,annotations}
   # 将您的图像复制到 data/custom/images/
   # 将您的标注复制到 data/custom/annotations/
   ```

2. **如果需要转换标注**：
   ```bash
   python main.py convert --mode xml2yolo --input data/custom/annotations/ --images data/custom/images/ --output data/custom/yolo_labels/
   ```

3. **准备数据集**：
   ```bash
   python main.py prepare --images data/custom/images/ --annotations data/custom/yolo_labels/ --output data/custom/dataset/
   ```

4. **训练模型**：
   ```bash
   python main.py train --data data/custom/dataset/dataset.yaml --epochs 200 --model yolov8s
   ```

### 自定义计数线

您可以编程方式定义自定义计数线：

```python
from src.detection.counter import TrafficCounter
from src.detection.detector import YOLODetector

detector = YOLODetector()
counter = TrafficCounter(detector)

# 添加水平线
counter.add_horizontal_counting_line(y_position=400, image_width=1920, name="main_line")

# 添加垂直线  
counter.add_vertical_counting_line(x_position=960, image_height=1080, name="side_line")

# 添加自定义线
counter.add_counting_line(start_point=(100, 200), end_point=(1800, 800), name="diagonal_line")
```

## 📈 性能优化

### GPU 加速
- 确保正确安装 CUDA
- 为您的 GPU 内存使用适当的批次大小
- 考虑模型大小与准确性的权衡

### 模型选择
- **YOLOv8n**：最快，最低准确性
- **YOLOv8s**：平衡速度/准确性  
- **YOLOv8m**：更高准确性，更慢
- **YOLOv8l/x**：最佳准确性，最慢

### 处理提示
- 使用较低分辨率以加快处理
- 根据需要调整置信阈值
- 启用视频处理的 GPU 加速

## 🧪 测试

运行测试套件：
```bash
python -m pytest tests/ -v
```

运行特定测试：
```bash
python -m pytest tests/test_detector.py -v
python -m pytest tests/test_counter.py -v
```

## 🤝 贡献

1. Fork 仓库
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送分支 (`git push origin feature/amazing-feature`)
5. 打开 Pull Request

## 📝 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) 优秀的 YOLO 实现
- [Gradio](https://gradio.app/) 美丽的 web 界面框架
- [OpenCV](https://opencv.org/) 计算机视觉实用工具
- 开源社区的各种工具和库

## 📞 支持

- 📧 邮箱: support@example.com
- 🐛 问题: [GitHub Issues](https://github.com/your-repo/yolo-traffic-counter/issues)
- 📖 文档: [Wiki](https://github.com/your-repo/yolo-traffic-counter/wiki)

## 🗺️ 路线图

- [ ] 实时相机支持
- [ ] 多种计数线类型 (多边形，曲线)
- [ ] 高级分析 (速度估计，车辆分类)
- [ ] 云部署支持
- [ ] 移动应用集成
- [ ] API 端点集成

---

**❤️ 为交通分析和计算机视觉爱好者而作**