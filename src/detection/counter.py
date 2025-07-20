"""
Traffic counter with object tracking
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict, deque
import time
import math
from threading import Lock

from src.detection.detector import YOLODetector, Detection
from utils.config_loader import config
from utils.logger import logger


class Track:
    """Object tracking class"""
    
    def __init__(self, track_id: int, detection: Detection):
        self.track_id = track_id
        self.class_id = detection.class_id
        self.class_name = detection.class_name
        self.positions = deque(maxlen=config.get('counting.track_history', 30))
        self.confidences = deque(maxlen=config.get('counting.track_history', 30))
        self.last_seen = time.time()
        self.counted = False
        
        # Add initial detection
        self.update(detection)
    
    def update(self, detection: Detection) -> None:
        """Update track with new detection"""
        self.positions.append(detection.center)
        self.confidences.append(detection.confidence)
        self.last_seen = time.time()
    
    @property
    def current_position(self) -> Tuple[float, float]:
        """Get current position"""
        return self.positions[-1] if self.positions else (0, 0)
    
    @property
    def previous_position(self) -> Optional[Tuple[float, float]]:
        """Get previous position"""
        return self.positions[-2] if len(self.positions) >= 2 else None
    
    @property
    def velocity(self) -> Tuple[float, float]:
        """Calculate velocity vector"""
        if len(self.positions) < 2:
            return (0, 0)
        
        current = self.positions[-1]
        previous = self.positions[-2]
        
        return (current[0] - previous[0], current[1] - previous[1])
    
    @property
    def speed(self) -> float:
        """Calculate speed"""
        vx, vy = self.velocity
        return math.sqrt(vx * vx + vy * vy)
    
    @property
    def age(self) -> int:
        """Get track age (number of frames)"""
        return len(self.positions)
    
    @property
    def time_since_last_seen(self) -> float:
        """Time since last update"""
        return time.time() - self.last_seen
    
    def predict_next_position(self) -> Tuple[float, float]:
        """Predict next position based on velocity"""
        if len(self.positions) < 2:
            return self.current_position
        
        current_x, current_y = self.current_position
        vx, vy = self.velocity
        
        return (current_x + vx, current_y + vy)


class CountingLine:
    """Counting line for traffic counting"""
    
    def __init__(self, start_point: Tuple[int, int], end_point: Tuple[int, int], 
                 name: str = "counting_line"):
        self.start_point = start_point
        self.end_point = end_point
        self.name = name
        
        # Calculate line parameters
        self._calculate_line_params()
    
    def _calculate_line_params(self) -> None:
        """Calculate line equation parameters"""
        x1, y1 = self.start_point
        x2, y2 = self.end_point
        
        # Line equation: ax + by + c = 0
        self.a = y2 - y1
        self.b = x1 - x2
        self.c = x2 * y1 - x1 * y2
        
        # Normalize
        norm = math.sqrt(self.a * self.a + self.b * self.b)
        if norm > 0:
            self.a /= norm
            self.b /= norm
            self.c /= norm
    
    def distance_to_point(self, point: Tuple[float, float]) -> float:
        """Calculate distance from point to line"""
        x, y = point
        return abs(self.a * x + self.b * y + self.c)
    
    def point_side(self, point: Tuple[float, float]) -> int:
        """Determine which side of line the point is on"""
        x, y = point
        value = self.a * x + self.b * y + self.c
        return 1 if value > 0 else -1 if value < 0 else 0
    
    def has_crossed(self, prev_point: Tuple[float, float], 
                   curr_point: Tuple[float, float]) -> bool:
        """Check if trajectory crossed the line"""
        prev_side = self.point_side(prev_point)
        curr_side = self.point_side(curr_point)
        
        return prev_side != curr_side and prev_side != 0 and curr_side != 0
    
    def draw(self, image: np.ndarray, color: Tuple[int, int, int] = (0, 255, 255), 
             thickness: int = 3) -> None:
        """Draw counting line on image"""
        cv2.line(image, self.start_point, self.end_point, color, thickness)
        
        # Draw line name
        mid_x = (self.start_point[0] + self.end_point[0]) // 2
        mid_y = (self.start_point[1] + self.end_point[1]) // 2
        
        cv2.putText(image, self.name, (mid_x, mid_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


class TrafficCounter:
    """Traffic counter with object tracking"""
    
    def __init__(self, detector: YOLODetector = None):
        self.detector = detector or YOLODetector()
        self.counting_config = config.get_counting_config()
        
        # Tracking parameters
        self.max_distance = 50.0  # Maximum distance for track association
        self.max_age = 30  # Maximum frames without detection
        self.min_track_length = self.counting_config.get('min_track_length', 5)
        
        # State
        self.tracks = {}  # track_id -> Track
        self.next_track_id = 1
        self.counting_lines = []
        self.counts = defaultdict(int)  # class_name -> count
        self.lock = Lock()
        
        logger.info("Traffic Counter initialized")
    
    def add_counting_line(self, start_point: Tuple[int, int], end_point: Tuple[int, int],
                         name: str = None) -> None:
        """Add counting line"""
        if name is None:
            name = f"line_{len(self.counting_lines) + 1}"
        
        counting_line = CountingLine(start_point, end_point, name)
        self.counting_lines.append(counting_line)
        
        logger.info(f"Added counting line: {name} from {start_point} to {end_point}")
    
    def add_horizontal_counting_line(self, y_position: int, image_width: int, 
                                   name: str = None) -> None:
        """Add horizontal counting line"""
        start_point = (0, y_position)
        end_point = (image_width, y_position)
        self.add_counting_line(start_point, end_point, name)
    
    def add_vertical_counting_line(self, x_position: int, image_height: int,
                                 name: str = None) -> None:
        """Add vertical counting line"""
        start_point = (x_position, 0)
        end_point = (x_position, image_height)
        self.add_counting_line(start_point, end_point, name)
    
    def _calculate_distance(self, pos1: Tuple[float, float], 
                          pos2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points"""
        return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)
    
    def _associate_detections_to_tracks(self, detections: List[Detection]) -> Dict[int, Detection]:
        """Associate detections to existing tracks"""
        associations = {}
        used_detections = set()
        
        # Calculate distance matrix
        for track_id, track in self.tracks.items():
            if track.time_since_last_seen > 1.0:  # Skip old tracks
                continue
                
            predicted_pos = track.predict_next_position()
            best_detection_idx = None
            best_distance = float('inf')
            
            for i, detection in enumerate(detections):
                if i in used_detections:
                    continue
                
                # Only associate detections of the same class
                if detection.class_id != track.class_id:
                    continue
                
                distance = self._calculate_distance(predicted_pos, detection.center)
                
                if distance < self.max_distance and distance < best_distance:
                    best_distance = distance
                    best_detection_idx = i
            
            if best_detection_idx is not None:
                associations[track_id] = detections[best_detection_idx]
                used_detections.add(best_detection_idx)
        
        # Create new tracks for unassociated detections
        for i, detection in enumerate(detections):
            if i not in used_detections:
                new_track = Track(self.next_track_id, detection)
                self.tracks[self.next_track_id] = new_track
                associations[self.next_track_id] = detection
                self.next_track_id += 1
        
        return associations
    
    def _update_tracks(self, associations: Dict[int, Detection]) -> None:
        """Update tracks with associated detections"""
        current_time = time.time()
        
        # Update associated tracks
        for track_id, detection in associations.items():
            if track_id in self.tracks:
                self.tracks[track_id].update(detection)
        
        # Remove old tracks
        tracks_to_remove = []
        for track_id, track in self.tracks.items():
            if track.time_since_last_seen > self.max_age / 30.0:  # Convert frames to seconds
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
    
    def _check_line_crossings(self) -> None:
        """Check for line crossings and update counts"""
        for track in self.tracks.values():
            if track.counted or track.age < self.min_track_length:
                continue
            
            if track.previous_position is None:
                continue
            
            # Check each counting line
            for line in self.counting_lines:
                if line.has_crossed(track.previous_position, track.current_position):
                    # Object crossed the line
                    self.counts[track.class_name] += 1
                    track.counted = True
                    
                    logger.info(f"Object crossed line {line.name}: {track.class_name} "
                              f"(Total: {self.counts[track.class_name]})")
                    break
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process single frame
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            Tuple of (annotated_frame, frame_statistics)
        """
        with self.lock:
            # Check if detector is initialized
            if self.detector is None:
                error_message = "Detector not initialized"
                logger.error(error_message)
                # Add error text to frame
                cv2.putText(frame, error_message, (50, 50), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                return frame, {"error": error_message}
            
            # Run detection
            detections, _ = self.detector.detect(frame)
            
            # Associate detections to tracks
            associations = self._associate_detections_to_tracks(detections)
            
            # Update tracks
            self._update_tracks(associations)
            
            # Check line crossings
            self._check_line_crossings()
            
            # Draw visualizations
            annotated_frame = self._draw_frame(frame.copy())
            
            # Collect frame statistics
            frame_stats = {
                'counts': dict(self.counts),
                'num_detections': len(detections),
                'num_tracks': len(self.tracks),
                'track_ids': list(self.tracks.keys())
            }
            
            return annotated_frame, frame_stats
    
    def _draw_frame(self, frame: np.ndarray) -> np.ndarray:
        """Draw annotations on frame"""
        # Draw counting lines
        for line in self.counting_lines:
            line.draw(frame)
        
        # Draw tracks
        colors = {
            0: (0, 255, 0),    # person - green
            1: (255, 0, 0),    # vehicle - red
        }
        
        for track in self.tracks.values():
            if len(track.positions) < 2:
                continue
            
            color = colors.get(track.class_id, (0, 255, 255))
            
            # Draw track history
            if len(track.positions) > 1:
                points = np.array(track.positions, dtype=np.int32)
                cv2.polylines(frame, [points], False, color, 2)
            
            # Draw current position
            x, y = [int(coord) for coord in track.current_position]
            cv2.circle(frame, (x, y), 5, color, -1)
            
            # Draw track ID and class
            label = f"ID:{track.track_id} {track.class_name}"
            cv2.putText(frame, label, (x + 10, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw counts
        y_offset = 30
        for class_name, count in self.counts.items():
            text = f"{class_name}: {count}"
            cv2.putText(frame, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            y_offset += 30
        
        # Draw total count
        total_text = f"Total: {sum(self.counts.values())}"
        cv2.putText(frame, total_text, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
        
        return frame
    
    def process_video(self, video_path: str, output_path: str = None,
                     show_video: bool = False, save_stats: bool = True) -> Dict[str, Any]:
        """
        Process video for counting
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video
            show_video: Whether to display video
            save_stats: Whether to save counting statistics
            
        Returns:
            Processing statistics
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
        
        # Add default counting line if none exist
        if not self.counting_lines:
            line_y = int(height * config.get('counting.line_position', 0.5))
            self.add_horizontal_counting_line(line_y, width, "default_line")
        
        # Initialize video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Initialize statistics
        stats = {
            'video_path': video_path,
            'total_frames': total_frames,
            'processed_frames': 0,
            'processing_time': 0,
            'fps': 0,
            'final_counts': {},
            'frame_stats': []
        }
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Process frame
                annotated_frame, frame_stats = self.process_frame(frame)
                
                # Save frame statistics
                if save_stats:
                    frame_stats['frame_number'] = frame_count
                    frame_stats['timestamp'] = frame_count / fps
                    stats['frame_stats'].append(frame_stats)
                
                # Write frame
                if writer:
                    writer.write(annotated_frame)
                
                # Show frame
                if show_video:
                    cv2.imshow('Traffic Counter', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Update progress
                if frame_count % (total_frames // 20 + 1) == 0:
                    progress = (frame_count / total_frames) * 100
                    logger.info(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
                    logger.info(f"Current counts: {dict(self.counts)}")
        
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
        stats['final_counts'] = dict(self.counts)
        
        logger.info(f"Video processing completed:")
        logger.info(f"Processed {frame_count} frames in {processing_time:.2f}s")
        logger.info(f"Average FPS: {stats['fps']:.2f}")
        logger.info(f"Final counts: {stats['final_counts']}")
        
        # Save statistics
        if save_stats:
            import json
            stats_path = Path(video_path).stem + "_counting_stats.json"
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2)
            logger.info(f"Counting statistics saved to: {stats_path}")
        
        return stats
    
    def reset_counts(self) -> None:
        """Reset all counts and tracks"""
        with self.lock:
            self.counts.clear()
            self.tracks.clear()
            self.next_track_id = 1
        
        logger.info("Counts and tracks reset")
    
    def get_current_counts(self) -> Dict[str, int]:
        """Get current counts"""
        return dict(self.counts)
    
    def get_active_tracks_info(self) -> List[Dict[str, Any]]:
        """Get information about active tracks"""
        tracks_info = []
        
        for track in self.tracks.values():
            info = {
                'track_id': track.track_id,
                'class_name': track.class_name,
                'position': track.current_position,
                'velocity': track.velocity,
                'speed': track.speed,
                'age': track.age,
                'counted': track.counted
            }
            tracks_info.append(info)
        
        return tracks_info


def main():
    """Main function for traffic counting"""
    import argparse
    
    parser = argparse.ArgumentParser(description="YOLO Traffic Counter")
    parser.add_argument("--input", "-i", type=str, required=True,
                       help="Input video file")
    parser.add_argument("--output", "-o", type=str,
                       help="Output video file")
    parser.add_argument("--weights", "-w", type=str,
                       help="Model weights file path")
    parser.add_argument("--show", action="store_true",
                       help="Show video during processing")
    parser.add_argument("--line-y", type=int,
                       help="Y position of counting line (default: center)")
    
    args = parser.parse_args()
    
    # Initialize detector and counter
    detector = YOLODetector(args.weights)
    counter = TrafficCounter(detector)
    
    # Add counting line
    if args.line_y:
        # We need to get video dimensions first
        cap = cv2.VideoCapture(args.input)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap.release()
        counter.add_horizontal_counting_line(args.line_y, width)
    
    # Process video
    stats = counter.process_video(
        args.input,
        output_path=args.output,
        show_video=args.show,
        save_stats=True
    )
    
    print(f"Counting completed!")
    print(f"Final counts: {stats['final_counts']}")
    print(f"Processing FPS: {stats['fps']:.2f}")


if __name__ == "__main__":
    main()