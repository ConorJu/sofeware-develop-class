"""
Data annotation tool for YOLO Traffic Counter
"""
import cv2
import os
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from threading import Lock

from utils.config_loader import config
from utils.logger import logger


class BoundingBox:
    """Bounding box class"""
    
    def __init__(self, x1: int, y1: int, x2: int, y2: int, class_id: int, class_name: str):
        self.x1 = min(x1, x2)
        self.y1 = min(y1, y2)
        self.x2 = max(x1, x2)
        self.y2 = max(y1, y2)
        self.class_id = class_id
        self.class_name = class_name
    
    @property
    def width(self) -> int:
        return self.x2 - self.x1
    
    @property
    def height(self) -> int:
        return self.y2 - self.y1
    
    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2
    
    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2
    
    def to_yolo_format(self, img_width: int, img_height: int) -> Tuple[int, float, float, float, float]:
        """Convert to YOLO format: class_id, center_x, center_y, width, height (normalized)"""
        center_x_norm = self.center_x / img_width
        center_y_norm = self.center_y / img_height
        width_norm = self.width / img_width
        height_norm = self.height / img_height
        
        return self.class_id, center_x_norm, center_y_norm, width_norm, height_norm
    
    def to_xml_element(self) -> ET.Element:
        """Convert to XML element for Pascal VOC format"""
        obj_elem = ET.Element('object')
        
        name_elem = ET.SubElement(obj_elem, 'name')
        name_elem.text = self.class_name
        
        bndbox_elem = ET.SubElement(obj_elem, 'bndbox')
        
        xmin_elem = ET.SubElement(bndbox_elem, 'xmin')
        xmin_elem.text = str(self.x1)
        
        ymin_elem = ET.SubElement(bndbox_elem, 'ymin')
        ymin_elem.text = str(self.y1)
        
        xmax_elem = ET.SubElement(bndbox_elem, 'xmax')
        xmax_elem.text = str(self.x2)
        
        ymax_elem = ET.SubElement(bndbox_elem, 'ymax')
        ymax_elem.text = str(self.y2)
        
        return obj_elem


class ImageAnnotator:
    """Image annotation tool"""
    
    def __init__(self):
        self.classes = config.get_classes()
        self.current_class = 0
        self.current_image = None
        self.current_image_path = ""
        self.annotations = []
        self.drawing = False
        self.start_point = (0, 0)
        self.temp_box = None
        self.lock = Lock()
        
        # Colors for different classes
        self.colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        
        logger.info("Image Annotator initialized")
    
    def mouse_callback(self, event, x, y, flags, param):
        """Mouse callback for drawing bounding boxes"""
        with self.lock:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.drawing = True
                self.start_point = (x, y)
                
            elif event == cv2.EVENT_MOUSEMOVE:
                if self.drawing:
                    self.temp_box = (self.start_point[0], self.start_point[1], x, y)
                    
            elif event == cv2.EVENT_LBUTTONUP:
                self.drawing = False
                if abs(x - self.start_point[0]) > 10 and abs(y - self.start_point[1]) > 10:
                    # Create bounding box
                    bbox = BoundingBox(
                        self.start_point[0], self.start_point[1], x, y,
                        self.current_class, self.classes[self.current_class]
                    )
                    self.annotations.append(bbox)
                    logger.info(f"Added bounding box: {self.classes[self.current_class]} at ({bbox.x1}, {bbox.y1}, {bbox.x2}, {bbox.y2})")
                self.temp_box = None
    
    def draw_annotations(self, image: np.ndarray) -> np.ndarray:
        """Draw annotations on image"""
        img_copy = image.copy()
        
        # Draw existing annotations
        for bbox in self.annotations:
            color = self.colors[bbox.class_id % len(self.colors)]
            cv2.rectangle(img_copy, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2), color, 2)
            
            # Draw label
            label = f"{bbox.class_name}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(img_copy, (bbox.x1, bbox.y1 - label_size[1] - 10),
                         (bbox.x1 + label_size[0], bbox.y1), color, -1)
            cv2.putText(img_copy, label, (bbox.x1, bbox.y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw temporary box
        if self.temp_box:
            color = self.colors[self.current_class % len(self.colors)]
            cv2.rectangle(img_copy, (self.temp_box[0], self.temp_box[1]),
                         (self.temp_box[2], self.temp_box[3]), color, 2)
        
        return img_copy
    
    def annotate_image(self, image_path: str) -> None:
        """Annotate a single image"""
        self.current_image_path = image_path
        self.current_image = cv2.imread(image_path)
        self.annotations = []
        
        if self.current_image is None:
            logger.error(f"Could not load image: {image_path}")
            return
        
        # Load existing annotations if they exist
        self.load_annotations()
        
        window_name = f"Annotate: {Path(image_path).name}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        logger.info(f"Annotating image: {image_path}")
        logger.info("Controls:")
        logger.info("- Left click and drag to draw bounding box")
        logger.info("- Press 0-9 to change class")
        logger.info("- Press 's' to save")
        logger.info("- Press 'd' to delete last annotation")
        logger.info("- Press 'c' to clear all annotations")
        logger.info("- Press 'q' to quit")
        
        while True:
            # Draw image with annotations
            display_img = self.draw_annotations(self.current_image)
            
            # Add info text
            info_text = f"Class: {self.current_class} ({self.classes[self.current_class]}) | Annotations: {len(self.annotations)}"
            cv2.putText(display_img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow(window_name, display_img)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save_annotations()
                logger.info("Annotations saved")
            elif key == ord('d'):
                if self.annotations:
                    removed = self.annotations.pop()
                    logger.info(f"Removed annotation: {removed.class_name}")
            elif key == ord('c'):
                self.annotations = []
                logger.info("All annotations cleared")
            elif key >= ord('0') and key <= ord('9'):
                class_id = key - ord('0')
                if class_id in self.classes:
                    self.current_class = class_id
                    logger.info(f"Changed to class: {self.classes[class_id]}")
        
        cv2.destroyWindow(window_name)
    
    def load_annotations(self) -> None:
        """Load existing annotations for current image"""
        annotation_path = self.get_annotation_path(format='xml')
        
        if Path(annotation_path).exists():
            try:
                tree = ET.parse(annotation_path)
                root = tree.getroot()
                
                for obj in root.findall('object'):
                    name = obj.find('name').text
                    class_id = next((k for k, v in self.classes.items() if v == name), 0)
                    
                    bndbox = obj.find('bndbox')
                    x1 = int(bndbox.find('xmin').text)
                    y1 = int(bndbox.find('ymin').text)
                    x2 = int(bndbox.find('xmax').text)
                    y2 = int(bndbox.find('ymax').text)
                    
                    bbox = BoundingBox(x1, y1, x2, y2, class_id, name)
                    self.annotations.append(bbox)
                
                logger.info(f"Loaded {len(self.annotations)} existing annotations")
            except Exception as e:
                logger.error(f"Error loading annotations: {e}")
    
    def save_annotations(self) -> None:
        """Save annotations in both XML and TXT formats"""
        if not self.annotations:
            return
        
        # Save as XML (Pascal VOC format)
        self.save_xml_annotations()
        
        # Save as TXT (YOLO format)
        self.save_txt_annotations()
    
    def save_xml_annotations(self) -> None:
        """Save annotations in XML format"""
        annotation_path = self.get_annotation_path(format='xml')
        
        # Create XML structure
        annotation = ET.Element('annotation')
        
        # Add filename
        filename_elem = ET.SubElement(annotation, 'filename')
        filename_elem.text = Path(self.current_image_path).name
        
        # Add image size
        size_elem = ET.SubElement(annotation, 'size')
        height, width, depth = self.current_image.shape
        
        width_elem = ET.SubElement(size_elem, 'width')
        width_elem.text = str(width)
        
        height_elem = ET.SubElement(size_elem, 'height')
        height_elem.text = str(height)
        
        depth_elem = ET.SubElement(size_elem, 'depth')
        depth_elem.text = str(depth)
        
        # Add objects
        for bbox in self.annotations:
            annotation.append(bbox.to_xml_element())
        
        # Save XML file
        tree = ET.ElementTree(annotation)
        tree.write(annotation_path, encoding='utf-8', xml_declaration=True)
    
    def save_txt_annotations(self) -> None:
        """Save annotations in YOLO TXT format"""
        annotation_path = self.get_annotation_path(format='txt')
        
        height, width, _ = self.current_image.shape
        
        with open(annotation_path, 'w') as f:
            for bbox in self.annotations:
                yolo_format = bbox.to_yolo_format(width, height)
                line = ' '.join(map(str, yolo_format))
                f.write(line + '\n')
    
    def get_annotation_path(self, format: str = 'xml') -> str:
        """Get annotation file path"""
        annotations_dir = Path(config.get('data.annotations_dir'))
        image_name = Path(self.current_image_path).stem
        
        if format == 'xml':
            return str(annotations_dir / f"{image_name}.xml")
        elif format == 'txt':
            return str(annotations_dir / f"{image_name}.txt")
    
    def annotate_directory(self, directory: str) -> None:
        """Annotate all images in a directory"""
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(Path(directory).glob(f"*{ext}"))
            image_files.extend(Path(directory).glob(f"*{ext.upper()}"))
        
        logger.info(f"Found {len(image_files)} images to annotate")
        
        for i, image_file in enumerate(sorted(image_files)):
            logger.info(f"Annotating image {i+1}/{len(image_files)}: {image_file.name}")
            self.annotate_image(str(image_file))


def main():
    """Main function for annotation tool"""
    import argparse
    
    parser = argparse.ArgumentParser(description="YOLO Traffic Counter Annotation Tool")
    parser.add_argument("--input", "-i", type=str, required=True,
                       help="Input image file or directory")
    
    args = parser.parse_args()
    
    annotator = ImageAnnotator()
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        annotator.annotate_image(str(input_path))
    elif input_path.is_dir():
        annotator.annotate_directory(str(input_path))
    else:
        logger.error(f"Invalid input path: {input_path}")


if __name__ == "__main__":
    main()