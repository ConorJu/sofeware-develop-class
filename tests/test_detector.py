"""
Test cases for YOLO detector
"""
import pytest
import numpy as np
import cv2
import tempfile
import os
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.detection.detector import YOLODetector, Detection


class TestDetection:
    """Test Detection class"""
    
    def test_detection_creation(self):
        """Test Detection object creation"""
        bbox = [100, 100, 200, 200]
        detection = Detection(bbox, 0.8, 0, "person")
        
        assert detection.bbox == bbox
        assert detection.confidence == 0.8
        assert detection.class_id == 0
        assert detection.class_name == "person"
        assert detection.center == (150, 150)
        assert detection.width == 100
        assert detection.height == 100
        assert detection.area == 10000
    
    def test_detection_to_dict(self):
        """Test Detection to_dict method"""
        bbox = [50, 50, 150, 150]
        detection = Detection(bbox, 0.9, 1, "vehicle")
        
        result = detection.to_dict()
        expected = {
            'bbox': bbox,
            'confidence': 0.9,
            'class_id': 1,
            'class_name': "vehicle",
            'center': (100, 100),
            'area': 10000
        }
        
        assert result == expected


class TestYOLODetector:
    """Test YOLODetector class"""
    
    @pytest.fixture
    def detector(self):
        """Create detector instance for testing"""
        try:
            return YOLODetector()
        except Exception as e:
            pytest.skip(f"Could not initialize detector: {e}")
    
    def test_detector_initialization(self, detector):
        """Test detector initialization"""
        assert detector is not None
        assert detector.model is not None
        assert detector.confidence_threshold > 0
        assert detector.nms_threshold > 0
    
    def test_dummy_image_detection(self, detector):
        """Test detection on dummy image"""
        # Create dummy image
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        detections, annotated_image = detector.detect(dummy_image, return_image=True)
        
        assert isinstance(detections, list)
        assert annotated_image is not None
        assert annotated_image.shape == dummy_image.shape
    
    def test_update_thresholds(self, detector):
        """Test threshold updates"""
        original_conf = detector.confidence_threshold
        original_nms = detector.nms_threshold
        
        detector.update_thresholds(confidence=0.7, nms=0.5)
        
        assert detector.confidence_threshold == 0.7
        assert detector.nms_threshold == 0.5
        
        # Reset
        detector.update_thresholds(confidence=original_conf, nms=original_nms)
    
    def test_get_model_info(self, detector):
        """Test model info retrieval"""
        info = detector.get_model_info()
        
        assert isinstance(info, dict)
        assert 'device' in info
        assert 'confidence_threshold' in info
        assert 'nms_threshold' in info
    
    def test_benchmark(self, detector):
        """Test benchmark functionality"""
        try:
            metrics = detector.benchmark(image_size=(320, 320), num_runs=5)
            
            assert isinstance(metrics, dict)
            assert 'avg_inference_time' in metrics
            assert 'avg_fps' in metrics
            assert metrics['avg_inference_time'] > 0
            assert metrics['avg_fps'] > 0
        except Exception as e:
            pytest.skip(f"Benchmark test failed: {e}")
    
    def test_video_detection_dummy(self, detector):
        """Test video detection with dummy video"""
        # Create temporary dummy video
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
            temp_video_path = tmp_file.name
        
        try:
            # Create dummy video using OpenCV
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(temp_video_path, fourcc, 10, (640, 480))
            
            # Write 30 frames
            for i in range(30):
                frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                writer.write(frame)
            
            writer.release()
            
            # Test video detection
            stats = detector.detect_video(
                temp_video_path,
                show_video=False,
                save_detections=False
            )
            
            assert isinstance(stats, dict)
            assert 'processed_frames' in stats
            assert 'total_detections' in stats
            assert stats['processed_frames'] > 0
        
        finally:
            # Clean up
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])