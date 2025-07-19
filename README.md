# 🚗 YOLO Traffic Counter

A comprehensive traffic analysis system using YOLO object detection for counting vehicles and pedestrians with real-time visualization and advanced tracking capabilities.

## ✨ Features

### 🎯 Core Functionality
- **Real-time Traffic Counting**: Count vehicles and pedestrians crossing designated lines
- **Object Tracking**: Advanced multi-object tracking with trajectory analysis
- **Video Processing**: Process various video formats (MP4, AVI, MOV, MKV)
- **Customizable Detection**: Adjustable confidence thresholds and counting lines

### 🎓 Model Training
- **Custom Model Training**: Train YOLO models on your own datasets
- **Multiple Model Sizes**: Support for YOLOv8n/s/m/l/x variants
- **Automated Dataset Preparation**: Easy dataset splitting and organization
- **Training Progress Monitoring**: Real-time training metrics and visualization

### 📝 Data Management
- **Image Annotation Tool**: Interactive GUI for labeling objects
- **Format Conversion**: Convert between XML (Pascal VOC) and TXT (YOLO) formats
- **Data Validation**: Automatic validation of annotation files
- **Batch Processing**: Process multiple files simultaneously

### 🌐 Web Interface
- **Modern UI**: Beautiful Gradio-based web interface
- **Real-time Analytics**: Interactive charts and visualizations
- **Progress Tracking**: Live processing status and statistics
- **Multi-tab Interface**: Organized workflow for different tasks

## 🚀 Quick Start

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-repo/yolo-traffic-counter.git
cd yolo-traffic-counter
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Launch the web interface**
```bash
python main.py web
```

Navigate to `http://localhost:8501` to access the web interface.

## 📋 Requirements

- Python 3.8+
- CUDA-capable GPU (recommended for training and inference)
- 8GB+ RAM
- OpenCV-compatible system

### Dependencies
- PyTorch >= 1.9.0
- Ultralytics YOLO >= 8.0.0
- OpenCV >= 4.5.0
- Gradio >= 3.0.0
- Plotly >= 5.0.0
- And more (see requirements.txt)

## 🎯 Usage

### Command Line Interface

The project provides a unified CLI for all operations:

#### 1. Launch Web Interface
```bash
# Basic launch
python main.py web

# Custom host and port
python main.py web --host 0.0.0.0 --port 8080

# Public sharing (creates public URL)
python main.py web --share
```

#### 2. Count Traffic in Video
```bash
# Basic counting
python main.py count --input video.mp4 --output result.mp4

# With custom settings
python main.py count --input video.mp4 --output result.mp4 --conf 0.6 --line-y 400

# Show video during processing
python main.py count --input video.mp4 --show
```

#### 3. Train Custom Model
```bash
# Train with dataset
python main.py train --data dataset.yaml --epochs 100 --batch 16

# Use larger model
python main.py train --data dataset.yaml --model yolov8m --epochs 200
```

#### 4. Annotate Images
```bash
# Annotate single image
python main.py annotate --input image.jpg

# Annotate directory of images
python main.py annotate --input images/
```

#### 5. Convert Annotation Formats
```bash
# XML to YOLO format
python main.py convert --mode xml2yolo --input annotations/ --output yolo_labels/

# YOLO to XML format
python main.py convert --mode yolo2xml --input yolo_labels/ --images images/ --output xml_annotations/

# Validate annotations
python main.py convert --mode validate --input annotations/ --output validation_report/
```

#### 6. Run Object Detection
```bash
# Detect objects in image
python main.py detect --input image.jpg --output result.jpg --show

# Process video
python main.py detect --input video.mp4 --output detected.mp4

# Benchmark performance
python main.py detect --input test.jpg --benchmark
```

#### 7. Prepare Dataset
```bash
python main.py prepare --images images/ --annotations annotations/ --output data/dataset --split 0.8 0.2
```

### Web Interface Usage

1. **Traffic Counting Tab**:
   - Upload video file
   - Adjust confidence threshold and counting line position
   - Click "Process Video" to start counting
   - View results with interactive charts

2. **Model Training Tab**:
   - Specify images and annotations folders
   - Configure training parameters
   - Start training process
   - Monitor progress

3. **Data Management Tab**:
   - Convert between annotation formats
   - Validate annotation files
   - Batch process multiple files

4. **Model Management Tab**:
   - Initialize models with custom weights
   - View model information and statistics

## 📊 Configuration

The system uses YAML configuration files located in `configs/config.yaml`. Key settings include:

```yaml
# Model Configuration
model:
  name: "yolov8n"
  num_classes: 2
  input_size: 640

# Training Configuration  
training:
  epochs: 100
  batch_size: 16
  learning_rate: 0.001

# Detection Configuration
detection:
  confidence_threshold: 0.5
  nms_threshold: 0.4

# Counting Configuration
counting:
  line_position: 0.5
  track_history: 30
  min_track_length: 5
```

## 🏗️ Project Structure

```
yolo-traffic-counter/
├── configs/                 # Configuration files
│   └── config.yaml
├── data/                    # Data directories
│   ├── raw/                # Raw input data
│   ├── processed/          # Processed data
│   ├── annotations/        # Annotation files
│   └── dataset/           # Prepared datasets
├── models/                 # Model weights
├── src/                   # Source code
│   ├── annotation/        # Data annotation tools
│   ├── detection/         # Object detection and counting
│   ├── training/          # Model training
│   └── frontend/          # Web interface
├── utils/                 # Utility functions
├── tests/                 # Test files
├── runs/                  # Training runs and results
├── logs/                  # Log files
├── main.py               # Main entry point
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 🔧 Advanced Usage

### Custom Model Training

1. **Prepare your dataset**:
   ```bash
   # Organize images and annotations
   mkdir -p data/custom/{images,annotations}
   # Copy your images to data/custom/images/
   # Copy your annotations to data/custom/annotations/
   ```

2. **Convert annotations if needed**:
   ```bash
   python main.py convert --mode xml2yolo --input data/custom/annotations/ --images data/custom/images/ --output data/custom/yolo_labels/
   ```

3. **Prepare dataset**:
   ```bash
   python main.py prepare --images data/custom/images/ --annotations data/custom/yolo_labels/ --output data/custom/dataset/
   ```

4. **Train model**:
   ```bash
   python main.py train --data data/custom/dataset/dataset.yaml --epochs 200 --model yolov8s
   ```

### Custom Counting Lines

You can define custom counting lines programmatically:

```python
from src.detection.counter import TrafficCounter
from src.detection.detector import YOLODetector

detector = YOLODetector()
counter = TrafficCounter(detector)

# Add horizontal line
counter.add_horizontal_counting_line(y_position=400, image_width=1920, name="main_line")

# Add vertical line  
counter.add_vertical_counting_line(x_position=960, image_height=1080, name="side_line")

# Add custom line
counter.add_counting_line(start_point=(100, 200), end_point=(1800, 800), name="diagonal_line")
```

## 📈 Performance Optimization

### GPU Acceleration
- Ensure CUDA is properly installed
- Use appropriate batch sizes for your GPU memory
- Consider model size vs accuracy trade-offs

### Model Selection
- **YOLOv8n**: Fastest, lowest accuracy
- **YOLOv8s**: Balanced speed/accuracy  
- **YOLOv8m**: Higher accuracy, slower
- **YOLOv8l/x**: Best accuracy, slowest

### Processing Tips
- Use lower resolution for faster processing
- Adjust confidence thresholds based on your needs
- Enable GPU acceleration for video processing

## 🧪 Testing

Run the test suite:
```bash
python -m pytest tests/ -v
```

Run specific tests:
```bash
python -m pytest tests/test_detector.py -v
python -m pytest tests/test_counter.py -v
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) for the excellent YOLO implementation
- [Gradio](https://gradio.app/) for the beautiful web interface framework
- [OpenCV](https://opencv.org/) for computer vision utilities
- The open-source community for various tools and libraries

## 📞 Support

- 📧 Email: support@example.com
- 🐛 Issues: [GitHub Issues](https://github.com/your-repo/yolo-traffic-counter/issues)
- 📖 Documentation: [Wiki](https://github.com/your-repo/yolo-traffic-counter/wiki)

## 🗺️ Roadmap

- [ ] Real-time camera support
- [ ] Multiple counting line types (polygonal, curved)
- [ ] Advanced analytics (speed estimation, vehicle classification)
- [ ] Cloud deployment support
- [ ] Mobile app integration
- [ ] API endpoints for integration

---

**Made with ❤️ for traffic analysis and computer vision enthusiasts**