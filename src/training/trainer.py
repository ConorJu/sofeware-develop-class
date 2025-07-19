"""
YOLO model trainer for Traffic Counter
"""
import os
import torch
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
from ultralytics import YOLO
from ultralytics.utils import callbacks
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import shutil

from utils.config_loader import config
from utils.logger import logger


class YOLOTrainer:
    """YOLO model trainer class"""
    
    def __init__(self, model_name: str = None):
        self.model_config = config.get_model_config()
        self.training_config = config.get_training_config()
        self.data_config = config.get_data_config()
        
        self.model_name = model_name or self.model_config.get('name', 'yolov8n')
        self.model = None
        self.device = self._get_device()
        
        # Create runs directory
        self.runs_dir = Path("runs")
        self.runs_dir.mkdir(exist_ok=True)
        
        logger.info(f"YOLO Trainer initialized with model: {self.model_name}")
        logger.info(f"Using device: {self.device}")
    
    def _get_device(self) -> str:
        """Get training device"""
        device_config = self.training_config.get('device', 'auto')
        
        if device_config == 'auto':
            if torch.cuda.is_available():
                return 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return 'mps'
            else:
                return 'cpu'
        else:
            return device_config
    
    def load_model(self, weights_path: str = None, pretrained: bool = True) -> None:
        """Load YOLO model"""
        try:
            if weights_path and Path(weights_path).exists():
                logger.info(f"Loading model from weights: {weights_path}")
                self.model = YOLO(weights_path)
            elif pretrained:
                logger.info(f"Loading pretrained model: {self.model_name}")
                self.model = YOLO(f"{self.model_name}.pt")
            else:
                logger.info(f"Creating new model: {self.model_name}")
                # Create model from scratch
                model_yaml = self._create_model_yaml()
                self.model = YOLO(model_yaml)
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _create_model_yaml(self) -> str:
        """Create model YAML configuration"""
        model_yaml_path = f"configs/{self.model_name}.yaml"
        
        # Basic YOLOv8 configuration
        model_config = {
            'nc': self.model_config.get('num_classes', 2),
            'depth_multiple': 0.33,
            'width_multiple': 0.25,
            'backbone': [
                [-1, 1, 'Conv', [64, 6, 2, 2]],
                [-1, 1, 'Conv', [128, 3, 2]],
                [-1, 3, 'C2f', [128, True]],
                [-1, 1, 'Conv', [256, 3, 2]],
                [-1, 6, 'C2f', [256, True]],
                [-1, 1, 'Conv', [512, 3, 2]],
                [-1, 6, 'C2f', [512, True]],
                [-1, 1, 'Conv', [1024, 3, 2]],
                [-1, 3, 'C2f', [1024, True]],
                [-1, 1, 'SPPF', [1024, 5]],
            ],
            'head': [
                [-1, 1, 'nn.Upsample', ['None', 2, 'nearest']],
                [[-1, 6], 1, 'Concat', [1]],
                [-1, 3, 'C2f', [512]],
                [-1, 1, 'nn.Upsample', ['None', 2, 'nearest']],
                [[-1, 4], 1, 'Concat', [1]],
                [-1, 3, 'C2f', [256]],
                [-1, 1, 'Conv', [256, 3, 1]],
                [-1, 1, 'nn.Upsample', ['None', 2, 'nearest']],
                [[-1, 2], 1, 'Concat', [1]],
                [-1, 3, 'C2f', [128]],
                [-1, 1, 'Conv', [128, 3, 1]],
                [[15, 18, 21], 1, 'Detect', ['nc', 'anchors']],
            ]
        }
        
        with open(model_yaml_path, 'w') as f:
            yaml.dump(model_config, f, default_flow_style=False)
        
        return model_yaml_path
    
    def prepare_dataset(self, images_dir: str, annotations_dir: str, split_ratio: Tuple[float, float] = None) -> str:
        """Prepare dataset for training"""
        if split_ratio is None:
            train_ratio = self.data_config.get('train_split', 0.8)
            val_ratio = self.data_config.get('val_split', 0.2)
        else:
            train_ratio, val_ratio = split_ratio
        
        images_dir = Path(images_dir)
        annotations_dir = Path(annotations_dir)
        
        # Create dataset directory structure
        dataset_dir = Path("data/dataset")
        train_images_dir = dataset_dir / "images" / "train"
        val_images_dir = dataset_dir / "images" / "val"
        train_labels_dir = dataset_dir / "labels" / "train"
        val_labels_dir = dataset_dir / "labels" / "val"
        
        for dir_path in [train_images_dir, val_images_dir, train_labels_dir, val_labels_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(images_dir.glob(f"*{ext}"))
            image_files.extend(images_dir.glob(f"*{ext.upper()}"))
        
        if not image_files:
            raise ValueError(f"No image files found in {images_dir}")
        
        # Shuffle and split
        np.random.seed(42)
        image_files = list(image_files)
        np.random.shuffle(image_files)
        
        split_idx = int(len(image_files) * train_ratio)
        train_files = image_files[:split_idx]
        val_files = image_files[split_idx:]
        
        logger.info(f"Dataset split: {len(train_files)} train, {len(val_files)} validation")
        
        # Copy files
        def copy_files(file_list, img_dest, label_dest):
            for img_file in file_list:
                # Copy image
                shutil.copy2(img_file, img_dest / img_file.name)
                
                # Copy corresponding label file
                label_file = annotations_dir / f"{img_file.stem}.txt"
                if label_file.exists():
                    shutil.copy2(label_file, label_dest / f"{img_file.stem}.txt")
                else:
                    # Create empty label file
                    (label_dest / f"{img_file.stem}.txt").touch()
        
        copy_files(train_files, train_images_dir, train_labels_dir)
        copy_files(val_files, val_images_dir, val_labels_dir)
        
        # Create dataset YAML
        dataset_yaml_path = dataset_dir / "dataset.yaml"
        dataset_config = {
            'path': str(dataset_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'nc': len(config.get_classes()),
            'names': list(config.get_classes().values())
        }
        
        with open(dataset_yaml_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        logger.info(f"Dataset prepared at: {dataset_dir}")
        return str(dataset_yaml_path)
    
    def train(self, data_yaml: str, save_dir: str = None) -> Dict[str, Any]:
        """Train the YOLO model"""
        if self.model is None:
            self.load_model(pretrained=self.model_config.get('pretrained', True))
        
        if save_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = self.runs_dir / f"train_{timestamp}"
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Training parameters
        train_params = {
            'data': data_yaml,
            'epochs': self.training_config.get('epochs', 100),
            'batch': self.training_config.get('batch_size', 16),
            'imgsz': self.model_config.get('input_size', 640),
            'device': self.device,
            'workers': self.training_config.get('workers', 4),
            'project': str(save_dir.parent),
            'name': save_dir.name,
            'exist_ok': True,
            'patience': self.training_config.get('patience', 10),
            'save': True,
            'save_period': 10,
            'cache': False,
            'plots': True,
            'verbose': True
        }
        
        logger.info("Starting training...")
        logger.info(f"Training parameters: {train_params}")
        
        try:
            # Train the model
            results = self.model.train(**train_params)
            
            # Save best model to models directory
            best_weights = save_dir / "weights" / "best.pt"
            if best_weights.exists():
                models_dir = Path(config.get('paths.models_dir'))
                models_dir.mkdir(exist_ok=True)
                shutil.copy2(best_weights, models_dir / "best.pt")
                logger.info(f"Best model saved to: {models_dir / 'best.pt'}")
            
            # Generate training report
            self._generate_training_report(results, save_dir)
            
            logger.info("Training completed successfully!")
            return results
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def validate(self, data_yaml: str, weights_path: str = None) -> Dict[str, Any]:
        """Validate the model"""
        if weights_path:
            self.load_model(weights_path, pretrained=False)
        elif self.model is None:
            weights_path = config.get('paths.weights_file')
            if Path(weights_path).exists():
                self.load_model(weights_path, pretrained=False)
            else:
                logger.error("No model weights found for validation")
                return {}
        
        logger.info("Starting validation...")
        
        try:
            # Validate the model
            results = self.model.val(
                data=data_yaml,
                imgsz=self.model_config.get('input_size', 640),
                device=self.device,
                verbose=True
            )
            
            logger.info("Validation completed successfully!")
            return results
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            raise
    
    def _generate_training_report(self, results, save_dir: Path) -> None:
        """Generate training report with plots"""
        try:
            report_dir = save_dir / "report"
            report_dir.mkdir(exist_ok=True)
            
            # Create training summary
            summary = {
                'model': self.model_name,
                'device': self.device,
                'training_config': self.training_config,
                'model_config': self.model_config,
                'timestamp': datetime.now().isoformat()
            }
            
            # Save summary
            with open(report_dir / "training_summary.yaml", 'w') as f:
                yaml.dump(summary, f, default_flow_style=False)
            
            # Try to extract metrics from results
            if hasattr(results, 'results_dict'):
                metrics = results.results_dict
                
                # Save metrics
                with open(report_dir / "metrics.yaml", 'w') as f:
                    yaml.dump(metrics, f, default_flow_style=False)
            
            logger.info(f"Training report saved to: {report_dir}")
            
        except Exception as e:
            logger.warning(f"Could not generate training report: {e}")
    
    def export_model(self, weights_path: str = None, format: str = 'onnx') -> str:
        """Export model to different formats"""
        if weights_path:
            self.load_model(weights_path, pretrained=False)
        elif self.model is None:
            weights_path = config.get('paths.weights_file')
            if Path(weights_path).exists():
                self.load_model(weights_path, pretrained=False)
            else:
                logger.error("No model weights found for export")
                return ""
        
        logger.info(f"Exporting model to {format} format...")
        
        try:
            export_path = self.model.export(format=format)
            logger.info(f"Model exported to: {export_path}")
            return export_path
            
        except Exception as e:
            logger.error(f"Model export failed: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        if self.model is None:
            return {}
        
        try:
            info = {
                'model_name': self.model_name,
                'device': self.device,
                'parameters': sum(p.numel() for p in self.model.model.parameters()),
                'trainable_parameters': sum(p.numel() for p in self.model.model.parameters() if p.requires_grad),
                'model_size_mb': sum(p.numel() * p.element_size() for p in self.model.model.parameters()) / (1024 * 1024)
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Could not get model info: {e}")
            return {}


def main():
    """Main function for training"""
    import argparse
    
    parser = argparse.ArgumentParser(description="YOLO Traffic Counter Trainer")
    parser.add_argument("--mode", "-m", choices=['train', 'val', 'export'], required=True,
                       help="Operation mode")
    parser.add_argument("--data", "-d", type=str, required=True,
                       help="Dataset YAML file path")
    parser.add_argument("--weights", "-w", type=str,
                       help="Model weights file path")
    parser.add_argument("--model", type=str, default="yolov8n",
                       help="Model name (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)")
    parser.add_argument("--format", type=str, default="onnx",
                       help="Export format (onnx, torchscript, etc.)")
    
    args = parser.parse_args()
    
    trainer = YOLOTrainer(args.model)
    
    if args.mode == 'train':
        results = trainer.train(args.data)
        print("Training completed!")
        
    elif args.mode == 'val':
        results = trainer.validate(args.data, args.weights)
        print("Validation completed!")
        
    elif args.mode == 'export':
        export_path = trainer.export_model(args.weights, args.format)
        print(f"Model exported to: {export_path}")


if __name__ == "__main__":
    main()