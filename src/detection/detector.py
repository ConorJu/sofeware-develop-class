"""
YOLO detector for traffic counting
"""
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from ultralytics import YOLO
import time
from threading import Lock

from utils.config_loader import config
from utils.logger import logger


class Detection:
    """Detection result class"""
    
    def __init__(self, bbox: List[float], confidence: float, class_id: int, class_name: str):
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = class_name
        self.center = self._calculate_center()
        
    def _calculate_center(self) -> Tuple[float, float]:
        """Calculate center point of bounding box"""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]
    
    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]
    
    @property
    def area(self) -> float:
        return self.width * self.height
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert detection to dictionary"""
        return {
            'bbox': self.bbox,
            'confidence': self.confidence,
            'class_id': self.class_id,
            'class_name': self.class_name,
            'center': self.center,
            'area': self.area
        }


class YOLODetector:
    """YOLO-based object detector"""
    
    def __init__(self, weights_path: str = None, device: str = 'auto'):
        self.detection_config = config.get_detection_config()
        self.classes = config.get_classes()
        
        self.confidence_threshold = self.detection_config.get('confidence_threshold', 0.5)
        self.nms_threshold = self.detection_config.get('nms_threshold', 0.4)
        self.max_detections = self.detection_config.get('max_detections', 100)
        
        self.device = self._get_device(device)
        self.model = None
        self.lock = Lock()
        
        # Load model
        self._load_model(weights_path)
        
        logger.info(f"YOLO Detector initialized on device: {self.device}")
    
    def _get_device(self, device: str) -> str:
        """Get detection device"""
        if device == 'auto':
            if torch.cuda.is_available():
                return 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return 'mps'
            else:
                return 'cpu'
        return device
    
    def _load_model(self, weights_path: str = None) -> None:
        """Load YOLO model"""
        try:
            # First priority: Explicit path provided
            if weights_path and Path(weights_path).exists():
                logger.info(f"Loading model from provided path: {weights_path}")
                self.model = YOLO(weights_path)
                logger.info(f"Model loaded successfully from: {weights_path}")
                return
            
            # Second priority: Config default path
            default_weights = config.get('paths.weights_file')
            if Path(default_weights).exists():
                logger.info(f"Loading model from config path: {default_weights}")
                self.model = YOLO(default_weights)
                logger.info(f"Model loaded successfully from: {default_weights}")
                return
            
            # Third priority: Local model file with name from config
            model_name = config.get('model.name', 'yolov8n')
            model_path = f"{model_name}.pt"
            if Path(model_path).exists():
                logger.info(f"Loading local model: {model_path}")
                self.model = YOLO(model_path)
                logger.info(f"Model loaded successfully from: {model_path}")
                return
            
            # Fourth priority: Download pretrained model if not found locally
            logger.info(f"No local model found, attempting to download pretrained model: {model_name}")
            self.model = YOLO(f"{model_name}.pt")
            logger.info(f"Pretrained model {model_name} downloaded and loaded successfully")
            
            # Move model to device
            if self.device != 'cpu':
                self.model.to(self.device)
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise RuntimeError(f"Failed to load YOLO model: {str(e)}")
    
    def detect(self, image: np.ndarray, return_image: bool = False) -> Tuple[List[Detection], Optional[np.ndarray]]:
        """
        Detect objects in image
        
        Args:
            image: Input image as numpy array
            return_image: Whether to return annotated image
            
        Returns:
            Tuple of (detections, annotated_image)
        """
        with self.lock:
            try:
                start_time = time.time()
                
                # Run inference
                results = self.model(
                    image,
                    conf=self.confidence_threshold,
                    iou=self.nms_threshold,
                    max_det=self.max_detections,
                    device=self.device,
                    verbose=False
                )
                
                inference_time = time.time() - start_time
                
                # Parse results
                detections = []
                annotated_image = None
                
                if results and len(results) > 0:
                    result = results[0]
                    
                    if result.boxes is not None and len(result.boxes) > 0:
                        boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
                        confidences = result.boxes.conf.cpu().numpy()
                        class_ids = result.boxes.cls.cpu().numpy().astype(int)
                        
                        for i in range(len(boxes)):
                            class_id = class_ids[i]
                            class_name = self.classes.get(class_id, f"class_{class_id}")
                            
                            detection = Detection(
                                bbox=boxes[i].tolist(),
                                confidence=float(confidences[i]),
                                class_id=class_id,
                                class_name=class_name
                            )
                            detections.append(detection)
                    
                    # Create annotated image if requested
                    if return_image:
                        annotated_image = self._draw_detections(image.copy(), detections)
                
                logger.debug(f"Detected {len(detections)} objects in {inference_time:.3f}s")
                
                return detections, annotated_image
                
            except Exception as e:
                logger.error(f"Detection error: {e}")
                return [], image.copy() if return_image else None
    
    def _draw_detections(self, image: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Draw detections on image"""
        colors = {
            0: (0, 255, 0),    # person - green
            1: (255, 0, 0),    # vehicle - red
        }
        
        for detection in detections:
            x1, y1, x2, y2 = [int(coord) for coord in detection.bbox]
            color = colors.get(detection.class_id, (0, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{detection.class_name}: {detection.confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Background for label
            cv2.rectangle(image, (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0], y1), color, -1)
            
            # Text
            cv2.putText(image, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return image
    
    def detect_video(self, video_path: str, output_path: str = None, 
                    show_video: bool = False, save_detections: bool = False) -> Dict[str, Any]:
        """
        Detect objects in video
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video
            show_video: Whether to display video during processing
            save_detections: Whether to save detection results
            
        Returns:
            Dictionary with processing statistics
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return {}
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Processing video: {video_path}")
        logger.info(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Initialize video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Initialize statistics
        stats = {
            'total_frames': total_frames,
            'processed_frames': 0,
            'total_detections': 0,
            'class_counts': {class_name: 0 for class_name in self.classes.values()},
            'processing_time': 0,
            'fps': 0
        }
        
        all_detections = []
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Detect objects
                detections, annotated_frame = self.detect(frame, return_image=True)
                
                # Update statistics
                stats['total_detections'] += len(detections)
                for detection in detections:
                    stats['class_counts'][detection.class_name] += 1
                
                # Save detections if requested
                if save_detections:
                    frame_detections = {
                        'frame': frame_count,
                        'timestamp': frame_count / fps,
                        'detections': [det.to_dict() for det in detections]
                    }
                    all_detections.append(frame_detections)
                
                # Write frame
                if writer:
                    writer.write(annotated_frame)
                
                # Show frame
                if show_video:
                    cv2.imshow('YOLO Detection', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Update progress
                if frame_count % (total_frames // 20 + 1) == 0:
                    progress = (frame_count / total_frames) * 100
                    logger.info(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
        
        except KeyboardInterrupt:
            logger.info("Processing interrupted by user")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            if show_video:
                cv2.destroyAllWindows()
        
        # Calculate final statistics
        processing_time = time.time() - start_time
        stats['processed_frames'] = frame_count
        stats['processing_time'] = processing_time
        stats['fps'] = frame_count / processing_time if processing_time > 0 else 0
        
        logger.info(f"Video processing completed:")
        logger.info(f"Processed {frame_count} frames in {processing_time:.2f}s")
        logger.info(f"Average FPS: {stats['fps']:.2f}")
        logger.info(f"Total detections: {stats['total_detections']}")
        logger.info(f"Class counts: {stats['class_counts']}")
        
        # Save detection results
        if save_detections and all_detections:
            import json
            detections_path = Path(video_path).stem + "_detections.json"
            with open(detections_path, 'w') as f:
                json.dump(all_detections, f, indent=2)
            logger.info(f"Detection results saved to: {detections_path}")
        
        return stats
    
    def benchmark(self, image_size: Tuple[int, int] = (640, 640), num_runs: int = 100) -> Dict[str, float]:
        """
        Benchmark detection performance
        
        Args:
            image_size: Size of test images
            num_runs: Number of benchmark runs
            
        Returns:
            Performance metrics
        """
        logger.info(f"Benchmarking detector with {num_runs} runs on {image_size} images")
        
        # Create dummy image
        dummy_image = np.random.randint(0, 255, (*image_size[::-1], 3), dtype=np.uint8)
        
        # Warmup
        for _ in range(5):
            self.detect(dummy_image)
        
        # Benchmark
        times = []
        total_detections = 0
        
        for i in range(num_runs):
            start_time = time.time()
            detections, _ = self.detect(dummy_image)
            end_time = time.time()
            
            times.append(end_time - start_time)
            total_detections += len(detections)
            
            if (i + 1) % (num_runs // 10) == 0:
                logger.info(f"Benchmark progress: {((i + 1) / num_runs) * 100:.1f}%")
        
        # Calculate metrics
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        avg_fps = 1.0 / avg_time if avg_time > 0 else 0
        
        metrics = {
            'avg_inference_time': avg_time,
            'std_inference_time': std_time,
            'min_inference_time': min_time,
            'max_inference_time': max_time,
            'avg_fps': avg_fps,
            'total_detections': total_detections,
            'avg_detections_per_image': total_detections / num_runs
        }
        
        logger.info("Benchmark results:")
        logger.info(f"Average inference time: {avg_time * 1000:.2f}ms ± {std_time * 1000:.2f}ms")
        logger.info(f"Average FPS: {avg_fps:.2f}")
        logger.info(f"Min/Max time: {min_time * 1000:.2f}ms / {max_time * 1000:.2f}ms")
        
        return metrics
    
    def update_thresholds(self, confidence: float = None, nms: float = None) -> None:
        """Update detection thresholds"""
        if confidence is not None:
            self.confidence_threshold = confidence
            logger.info(f"Updated confidence threshold to: {confidence}")
        
        if nms is not None:
            self.nms_threshold = nms
            logger.info(f"Updated NMS threshold to: {nms}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        if self.model is None:
            return {}
        
        try:
            # Get model file information
            model_file = Path(self.model.ckpt_path) if hasattr(self.model, 'ckpt_path') else None
            model_size = model_file.stat().st_size // (1024 * 1024) if model_file and model_file.exists() else None
            
            # Get model details
            model_name = self.model.model.yaml_file.stem if hasattr(self.model, 'model') and hasattr(self.model.model, 'yaml_file') else "yolo"
            
            # Extract input size from model
            input_size = None
            try:
                if hasattr(self.model, 'model') and hasattr(self.model.model, 'args'):
                    input_size = self.model.model.args.get('imgsz', 640)
                else:
                    input_size = 640  # default YOLO size
            except:
                input_size = 640
            
            # Get task type
            task = "detect"
            if hasattr(self.model, 'task'):
                task = self.model.task
                
            info = {
                'name': model_name,
                'path': str(model_file) if model_file else "unknown",
                'size_mb': model_size,
                'model_type': type(self.model).__name__,
                'device': self.device,
                'input_size': input_size,
                'task': task,
                'confidence_threshold': self.confidence_threshold,
                'nms_threshold': self.nms_threshold,
                'max_detections': self.max_detections,
                'classes': self.classes,
                'num_classes': len(self.classes)
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Could not get complete model info: {str(e)}")
            
            # Return minimal info if detailed extraction fails
            return {
                'model_type': type(self.model).__name__,
                'device': self.device,
                'classes': self.classes,
                'confidence_threshold': self.confidence_threshold
            }


def main():
    """Main function for detection testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="YOLO Traffic Counter Detector")
    parser.add_argument("--input", "-i", type=str, required=True,
                       help="Input image or video file")
    parser.add_argument("--output", "-o", type=str,
                       help="Output file path")
    parser.add_argument("--weights", "-w", type=str,
                       help="Model weights file path")
    parser.add_argument("--conf", type=float, default=0.5,
                       help="Confidence threshold")
    parser.add_argument("--show", action="store_true",
                       help="Show detection results")
    parser.add_argument("--benchmark", action="store_true",
                       help="Run benchmark test")
    
    args = parser.parse_args()
    
    detector = YOLODetector(args.weights)
    detector.update_thresholds(confidence=args.conf)
    
    input_path = Path(args.input)
    
    if args.benchmark:
        detector.benchmark()
    elif input_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
        # Image detection
        image = cv2.imread(str(input_path))
        if image is None:
            logger.error(f"Could not load image: {input_path}")
            return
        
        detections, annotated_image = detector.detect(image, return_image=True)
        
        print(f"Detected {len(detections)} objects:")
        for detection in detections:
            print(f"  - {detection.class_name}: {detection.confidence:.3f}")
        
        if args.output:
            cv2.imwrite(args.output, annotated_image)
            logger.info(f"Result saved to: {args.output}")
        
        if args.show:
            cv2.imshow('Detection Result', annotated_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    elif input_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
        # Video detection
        stats = detector.detect_video(
            str(input_path),
            output_path=args.output,
            show_video=args.show,
            save_detections=True
        )
        
        print(f"Video processing completed:")
        print(f"Total detections: {stats['total_detections']}")
        print(f"Class counts: {stats['class_counts']}")
        print(f"Processing FPS: {stats['fps']:.2f}")


if __name__ == "__main__":
    main()