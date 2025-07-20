# 🚗 YOLO 交通计数器


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


1. **安装依赖**
```bash
pip install -r requirements.txt
```

2. **启动 web 界面**
```bash
python main.py web
```

导航到 `http://localhost:8502` 访问 web 界面。


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

