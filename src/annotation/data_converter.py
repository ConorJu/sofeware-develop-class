"""
Data format converter for YOLO Traffic Counter
Converts between XML (Pascal VOC) and TXT (YOLO) formats
"""
import xml.etree.ElementTree as ET
import os
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from utils.config_loader import config
from utils.logger import logger


class DataConverter:
    """Data format converter class"""
    
    def __init__(self):
        self.classes = config.get_classes()
        self.class_name_to_id = {v: k for k, v in self.classes.items()}
        logger.info("Data Converter initialized")
    
    def xml_to_yolo(self, xml_path: str, image_path: str = None) -> Optional[List[str]]:
        """Convert XML annotation to YOLO format"""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Get image dimensions
            size_elem = root.find('size')
            if size_elem is not None:
                img_width = int(size_elem.find('width').text)
                img_height = int(size_elem.find('height').text)
            else:
                # Try to get dimensions from image file
                if image_path and Path(image_path).exists():
                    img = cv2.imread(image_path)
                    if img is not None:
                        img_height, img_width = img.shape[:2]
                    else:
                        logger.error(f"Could not read image: {image_path}")
                        return None
                else:
                    logger.error(f"No image dimensions found in XML and no image path provided: {xml_path}")
                    return None
            
            yolo_lines = []
            
            # Process each object
            for obj in root.findall('object'):
                name = obj.find('name').text
                
                # Map class name to ID
                if name in self.class_name_to_id:
                    class_id = self.class_name_to_id[name]
                else:
                    logger.warning(f"Unknown class name: {name}, skipping...")
                    continue
                
                # Get bounding box coordinates
                bndbox = obj.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)
                
                # Convert to YOLO format (normalized center coordinates and dimensions)
                center_x = (xmin + xmax) / 2.0 / img_width
                center_y = (ymin + ymax) / 2.0 / img_height
                width = (xmax - xmin) / img_width
                height = (ymax - ymin) / img_height
                
                # Create YOLO format line
                yolo_line = f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"
                yolo_lines.append(yolo_line)
            
            return yolo_lines
            
        except Exception as e:
            logger.error(f"Error converting XML to YOLO: {xml_path}, Error: {e}")
            return None
    
    def yolo_to_xml(self, yolo_path: str, image_path: str, output_path: str = None) -> bool:
        """Convert YOLO annotation to XML format"""
        try:
            # Read image to get dimensions
            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"Could not read image: {image_path}")
                return False
            
            img_height, img_width, img_depth = img.shape
            
            # Create XML structure
            annotation = ET.Element('annotation')
            
            # Add filename
            filename_elem = ET.SubElement(annotation, 'filename')
            filename_elem.text = Path(image_path).name
            
            # Add image size
            size_elem = ET.SubElement(annotation, 'size')
            
            width_elem = ET.SubElement(size_elem, 'width')
            width_elem.text = str(img_width)
            
            height_elem = ET.SubElement(size_elem, 'height')
            height_elem.text = str(img_height)
            
            depth_elem = ET.SubElement(size_elem, 'depth')
            depth_elem.text = str(img_depth)
            
            # Read YOLO annotations
            with open(yolo_path, 'r') as f:
                lines = f.readlines()
            
            # Process each annotation
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) != 5:
                    continue
                
                class_id = int(parts[0])
                center_x = float(parts[1])
                center_y = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                # Convert to absolute coordinates
                abs_center_x = center_x * img_width
                abs_center_y = center_y * img_height
                abs_width = width * img_width
                abs_height = height * img_height
                
                xmin = int(abs_center_x - abs_width / 2)
                ymin = int(abs_center_y - abs_height / 2)
                xmax = int(abs_center_x + abs_width / 2)
                ymax = int(abs_center_y + abs_height / 2)
                
                # Create object element
                obj_elem = ET.SubElement(annotation, 'object')
                
                name_elem = ET.SubElement(obj_elem, 'name')
                name_elem.text = self.classes.get(class_id, f"class_{class_id}")
                
                bndbox_elem = ET.SubElement(obj_elem, 'bndbox')
                
                xmin_elem = ET.SubElement(bndbox_elem, 'xmin')
                xmin_elem.text = str(xmin)
                
                ymin_elem = ET.SubElement(bndbox_elem, 'ymin')
                ymin_elem.text = str(ymin)
                
                xmax_elem = ET.SubElement(bndbox_elem, 'xmax')
                xmax_elem.text = str(xmax)
                
                ymax_elem = ET.SubElement(bndbox_elem, 'ymax')
                ymax_elem.text = str(ymax)
            
            # Save XML file
            if output_path is None:
                output_path = yolo_path.replace('.txt', '.xml')
            
            tree = ET.ElementTree(annotation)
            tree.write(output_path, encoding='utf-8', xml_declaration=True)
            
            return True
            
        except Exception as e:
            logger.error(f"Error converting YOLO to XML: {yolo_path}, Error: {e}")
            return False
    
    def convert_xml_directory_to_yolo(self, xml_dir: str, output_dir: str, image_dir: str = None) -> None:
        """Convert all XML files in directory to YOLO format"""
        xml_dir = Path(xml_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        xml_files = list(xml_dir.glob("*.xml"))
        
        if not xml_files:
            logger.warning(f"No XML files found in {xml_dir}")
            return
        
        logger.info(f"Converting {len(xml_files)} XML files to YOLO format")
        
        def convert_single_xml(xml_file):
            """Convert single XML file"""
            try:
                # Find corresponding image file
                image_path = None
                if image_dir:
                    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
                    for ext in image_extensions:
                        potential_image = Path(image_dir) / f"{xml_file.stem}{ext}"
                        if potential_image.exists():
                            image_path = str(potential_image)
                            break
                
                yolo_lines = self.xml_to_yolo(str(xml_file), image_path)
                
                if yolo_lines is not None:
                    output_file = output_dir / f"{xml_file.stem}.txt"
                    with open(output_file, 'w') as f:
                        f.write('\n'.join(yolo_lines) + '\n')
                    
                    return f"Converted: {xml_file.name}"
                else:
                    return f"Failed: {xml_file.name}"
                    
            except Exception as e:
                return f"Error: {xml_file.name} - {e}"
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(convert_single_xml, xml_file) for xml_file in xml_files]
            
            success_count = 0
            for future in tqdm(as_completed(futures), total=len(xml_files), desc="Converting XML to YOLO"):
                result = future.result()
                if result.startswith("Converted"):
                    success_count += 1
                logger.debug(result)
        
        logger.info(f"Successfully converted {success_count}/{len(xml_files)} XML files to YOLO format")
    
    def convert_yolo_directory_to_xml(self, yolo_dir: str, image_dir: str, output_dir: str) -> None:
        """Convert all YOLO files in directory to XML format"""
        yolo_dir = Path(yolo_dir)
        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        txt_files = list(yolo_dir.glob("*.txt"))
        
        if not txt_files:
            logger.warning(f"No TXT files found in {yolo_dir}")
            return
        
        logger.info(f"Converting {len(txt_files)} YOLO files to XML format")
        
        def convert_single_yolo(txt_file):
            """Convert single YOLO file"""
            try:
                # Find corresponding image file
                image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
                image_path = None
                
                for ext in image_extensions:
                    potential_image = image_dir / f"{txt_file.stem}{ext}"
                    if potential_image.exists():
                        image_path = str(potential_image)
                        break
                
                if image_path is None:
                    return f"No image found for: {txt_file.name}"
                
                output_file = output_dir / f"{txt_file.stem}.xml"
                success = self.yolo_to_xml(str(txt_file), image_path, str(output_file))
                
                if success:
                    return f"Converted: {txt_file.name}"
                else:
                    return f"Failed: {txt_file.name}"
                    
            except Exception as e:
                return f"Error: {txt_file.name} - {e}"
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(convert_single_yolo, txt_file) for txt_file in txt_files]
            
            success_count = 0
            for future in tqdm(as_completed(futures), total=len(txt_files), desc="Converting YOLO to XML"):
                result = future.result()
                if result.startswith("Converted"):
                    success_count += 1
                logger.debug(result)
        
        logger.info(f"Successfully converted {success_count}/{len(txt_files)} YOLO files to XML format")
    
    def create_dataset_yaml(self, output_path: str, train_dir: str, val_dir: str) -> None:
        """Create dataset YAML file for YOLO training"""
        dataset_config = {
            'path': str(Path.cwd()),
            'train': train_dir,
            'val': val_dir,
            'nc': len(self.classes),
            'names': list(self.classes.values())
        }
        
        with open(output_path, 'w') as f:
            import yaml
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        logger.info(f"Created dataset YAML file: {output_path}")
    
    def validate_annotations(self, annotation_dir: str, image_dir: str = None) -> Dict[str, Any]:
        """Validate annotation files"""
        annotation_dir = Path(annotation_dir)
        results = {
            'total_files': 0,
            'valid_files': 0,
            'invalid_files': 0,
            'errors': []
        }
        
        # Check XML files
        xml_files = list(annotation_dir.glob("*.xml"))
        txt_files = list(annotation_dir.glob("*.txt"))
        
        logger.info(f"Validating {len(xml_files)} XML files and {len(txt_files)} TXT files")
        
        # Validate XML files
        for xml_file in xml_files:
            results['total_files'] += 1
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                
                # Check required elements
                if root.find('filename') is None:
                    results['errors'].append(f"{xml_file.name}: Missing filename element")
                    results['invalid_files'] += 1
                    continue
                
                if root.find('size') is None:
                    results['errors'].append(f"{xml_file.name}: Missing size element")
                    results['invalid_files'] += 1
                    continue
                
                objects = root.findall('object')
                if not objects:
                    results['errors'].append(f"{xml_file.name}: No objects found")
                    results['invalid_files'] += 1
                    continue
                
                # Validate each object
                valid_objects = 0
                for obj in objects:
                    name = obj.find('name')
                    bndbox = obj.find('bndbox')
                    
                    if name is None or bndbox is None:
                        continue
                    
                    if name.text not in self.class_name_to_id:
                        results['errors'].append(f"{xml_file.name}: Unknown class '{name.text}'")
                        continue
                    
                    # Check bounding box coordinates
                    try:
                        xmin = int(bndbox.find('xmin').text)
                        ymin = int(bndbox.find('ymin').text)
                        xmax = int(bndbox.find('xmax').text)
                        ymax = int(bndbox.find('ymax').text)
                        
                        if xmin >= xmax or ymin >= ymax:
                            results['errors'].append(f"{xml_file.name}: Invalid bounding box coordinates")
                            continue
                        
                        valid_objects += 1
                    except (ValueError, AttributeError):
                        results['errors'].append(f"{xml_file.name}: Invalid bounding box format")
                        continue
                
                if valid_objects > 0:
                    results['valid_files'] += 1
                else:
                    results['invalid_files'] += 1
                
            except Exception as e:
                results['errors'].append(f"{xml_file.name}: {str(e)}")
                results['invalid_files'] += 1
        
        # Validate TXT files
        for txt_file in txt_files:
            results['total_files'] += 1
            try:
                with open(txt_file, 'r') as f:
                    lines = f.readlines()
                
                valid_lines = 0
                for i, line in enumerate(lines):
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) != 5:
                        results['errors'].append(f"{txt_file.name}: Line {i+1} has wrong format")
                        continue
                    
                    try:
                        class_id = int(parts[0])
                        center_x = float(parts[1])
                        center_y = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        # Validate ranges
                        if class_id not in self.classes:
                            results['errors'].append(f"{txt_file.name}: Line {i+1} has invalid class ID")
                            continue
                        
                        if not (0 <= center_x <= 1 and 0 <= center_y <= 1 and 0 < width <= 1 and 0 < height <= 1):
                            results['errors'].append(f"{txt_file.name}: Line {i+1} has invalid coordinates")
                            continue
                        
                        valid_lines += 1
                    except ValueError:
                        results['errors'].append(f"{txt_file.name}: Line {i+1} has invalid number format")
                        continue
                
                if valid_lines > 0:
                    results['valid_files'] += 1
                else:
                    results['invalid_files'] += 1
                
            except Exception as e:
                results['errors'].append(f"{txt_file.name}: {str(e)}")
                results['invalid_files'] += 1
        
        logger.info(f"Validation complete: {results['valid_files']}/{results['total_files']} files are valid")
        if results['errors']:
            logger.warning(f"Found {len(results['errors'])} errors")
        
        return results


def main():
    """Main function for data converter"""
    import argparse
    
    parser = argparse.ArgumentParser(description="YOLO Traffic Counter Data Converter")
    parser.add_argument("--mode", "-m", choices=['xml2yolo', 'yolo2xml', 'validate'], required=True,
                       help="Conversion mode")
    parser.add_argument("--input", "-i", type=str, required=True,
                       help="Input directory")
    parser.add_argument("--output", "-o", type=str, required=True,
                       help="Output directory")
    parser.add_argument("--images", type=str,
                       help="Images directory (required for yolo2xml and validation)")
    
    args = parser.parse_args()
    
    converter = DataConverter()
    
    if args.mode == 'xml2yolo':
        converter.convert_xml_directory_to_yolo(args.input, args.output, args.images)
    elif args.mode == 'yolo2xml':
        if not args.images:
            logger.error("Images directory is required for yolo2xml mode")
            return
        converter.convert_yolo_directory_to_xml(args.input, args.images, args.output)
    elif args.mode == 'validate':
        results = converter.validate_annotations(args.input, args.images)
        print(f"Validation Results:")
        print(f"Total files: {results['total_files']}")
        print(f"Valid files: {results['valid_files']}")
        print(f"Invalid files: {results['invalid_files']}")
        if results['errors']:
            print(f"Errors:")
            for error in results['errors'][:10]:  # Show first 10 errors
                print(f"  - {error}")
            if len(results['errors']) > 10:
                print(f"  ... and {len(results['errors']) - 10} more errors")


if __name__ == "__main__":
    main()