#!/usr/bin/env python3
"""
Installation verification script for YOLO Traffic Counter
"""
import sys
import importlib
from pathlib import Path
import traceback

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

def check_python_version():
    """Check Python version"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor}.{version.micro} - Need Python 3.8+")
        return False

def check_dependencies():
    """Check required dependencies"""
    print("\n📦 Checking dependencies...")
    
    dependencies = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('ultralytics', 'Ultralytics YOLO'),
        ('cv2', 'OpenCV'),
        ('numpy', 'NumPy'),
        ('PIL', 'Pillow'),
        ('matplotlib', 'Matplotlib'),
        ('yaml', 'PyYAML'),
        ('tqdm', 'tqdm'),
        ('gradio', 'Gradio'),
        ('plotly', 'Plotly'),
        ('pandas', 'Pandas'),
    ]
    
    all_ok = True
    for module, name in dependencies:
        try:
            importlib.import_module(module)
            print(f"   ✅ {name} - OK")
        except ImportError as e:
            print(f"   ❌ {name} - Missing: {e}")
            all_ok = False
    
    return all_ok

def check_gpu_availability():
    """Check GPU availability"""
    print("\n🚀 Checking GPU availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"   ✅ CUDA available - {gpu_count} GPU(s)")
            print(f"   📱 GPU 0: {gpu_name}")
            return True
        else:
            print("   ⚠️  CUDA not available - will use CPU")
            return False
    except Exception as e:
        print(f"   ❌ Error checking GPU: {e}")
        return False

def check_config_loading():
    """Check configuration loading"""
    print("\n⚙️ Checking configuration...")
    
    try:
        from utils.config_loader import config
        model_config = config.get_model_config()
        print(f"   ✅ Config loaded - Model: {model_config.get('name', 'unknown')}")
        return True
    except Exception as e:
        print(f"   ❌ Config loading failed: {e}")
        traceback.print_exc()
        return False

def check_yolo_model():
    """Check YOLO model loading"""
    print("\n🎯 Checking YOLO model...")
    
    try:
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')  # This will download if not present
        print("   ✅ YOLO model loaded successfully")
        return True
    except Exception as e:
        print(f"   ❌ YOLO model loading failed: {e}")
        return False

def check_detector():
    """Check detector initialization"""
    print("\n🔍 Checking detector...")
    
    try:
        from src.detection.detector import YOLODetector
        detector = YOLODetector()
        print("   ✅ Detector initialized successfully")
        
        # Test with dummy image
        import numpy as np
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        detections, _ = detector.detect(dummy_image)
        print(f"   ✅ Detection test passed - Found {len(detections)} objects")
        return True
    except Exception as e:
        print(f"   ❌ Detector check failed: {e}")
        traceback.print_exc()
        return False

def check_counter():
    """Check traffic counter"""
    print("\n🚗 Checking traffic counter...")
    
    try:
        from src.detection.counter import TrafficCounter
        from src.detection.detector import YOLODetector
        
        detector = YOLODetector()
        counter = TrafficCounter(detector)
        print("   ✅ Traffic counter initialized successfully")
        return True
    except Exception as e:
        print(f"   ❌ Traffic counter check failed: {e}")
        traceback.print_exc()
        return False

def check_annotator():
    """Check annotation tool"""
    print("\n✏️ Checking annotation tool...")
    
    try:
        from src.annotation.annotator import ImageAnnotator
        annotator = ImageAnnotator()
        print("   ✅ Annotation tool initialized successfully")
        return True
    except Exception as e:
        print(f"   ❌ Annotation tool check failed: {e}")
        traceback.print_exc()
        return False

def check_data_converter():
    """Check data converter"""
    print("\n🔄 Checking data converter...")
    
    try:
        from src.annotation.data_converter import DataConverter
        converter = DataConverter()
        print("   ✅ Data converter initialized successfully")
        return True
    except Exception as e:
        print(f"   ❌ Data converter check failed: {e}")
        traceback.print_exc()
        return False

def check_trainer():
    """Check model trainer"""
    print("\n🎓 Checking model trainer...")
    
    try:
        from src.training.trainer import YOLOTrainer
        trainer = YOLOTrainer()
        print("   ✅ Model trainer initialized successfully")
        return True
    except Exception as e:
        print(f"   ❌ Model trainer check failed: {e}")
        traceback.print_exc()
        return False

def check_frontend():
    """Check frontend"""
    print("\n🌐 Checking frontend...")
    
    try:
        from src.frontend.app import TrafficCounterApp
        app = TrafficCounterApp()
        print("   ✅ Frontend app initialized successfully")
        return True
    except Exception as e:
        print(f"   ❌ Frontend check failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Main verification function"""
    print("🔧 YOLO Traffic Counter - Installation Verification")
    print("=" * 60)
    
    checks = [
        check_python_version,
        check_dependencies,
        check_gpu_availability,
        check_config_loading,
        check_yolo_model,
        check_detector,
        check_counter,
        check_annotator,
        check_data_converter,
        check_trainer,
        check_frontend,
    ]
    
    results = []
    for check in checks:
        try:
            result = check()
            results.append(result)
        except Exception as e:
            print(f"   ❌ Unexpected error in {check.__name__}: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"🎉 All checks passed! ({passed}/{total})")
        print("\n✅ Your installation is ready to use!")
        print("\nNext steps:")
        print("1. Run 'python main.py web' to launch the web interface")
        print("2. Or use 'python main.py --help' to see all available commands")
    else:
        print(f"⚠️  Some checks failed ({passed}/{total} passed)")
        print("\n❌ Please fix the issues above before using the system.")
        print("\nTroubleshooting tips:")
        print("1. Install missing dependencies: pip install -r requirements.txt")
        print("2. Check CUDA installation if GPU is needed")
        print("3. Ensure Python 3.8+ is being used")
        
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())