#!/usr/bin/env python3
"""
YOLO Traffic Counter - Main Entry Point
"""
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from utils.config_loader import config
from utils.logger import logger


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="YOLO Traffic Counter - Comprehensive Traffic Analysis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch web interface
  python main.py web --host 0.0.0.0 --port 8080
  
  # Count traffic in video
  python main.py count --input video.mp4 --output result.mp4
  
  # Train model
  python main.py train --data dataset.yaml --epochs 100
  
  # Annotate images
  python main.py annotate --input images/ 
  
  # Convert annotations
  python main.py convert --mode xml2yolo --input annotations/ --output yolo_labels/
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Web interface command
    web_parser = subparsers.add_parser('web', help='Launch web interface')
    web_parser.add_argument('--host', type=str, default='localhost',
                           help='Host address (default: localhost)')
    web_parser.add_argument('--port', type=int, default=8501,
                           help='Port number (default: 8501)')
    web_parser.add_argument('--share', action='store_true',
                           help='Create public link')
    web_parser.add_argument('--debug', action='store_true',
                           help='Enable debug mode')
    
    # Traffic counting command
    count_parser = subparsers.add_parser('count', help='Count traffic in video')
    count_parser.add_argument('--input', '-i', type=str, required=True,
                             help='Input video file')
    count_parser.add_argument('--output', '-o', type=str,
                             help='Output video file')
    count_parser.add_argument('--weights', '-w', type=str,
                             help='Model weights file')
    count_parser.add_argument('--conf', type=float, default=0.5,
                             help='Confidence threshold')
    count_parser.add_argument('--line-y', type=int,
                             help='Y position of counting line')
    count_parser.add_argument('--show', action='store_true',
                             help='Show video during processing')
    
    # Model training command
    train_parser = subparsers.add_parser('train', help='Train YOLO model')
    train_parser.add_argument('--data', '-d', type=str, required=True,
                             help='Dataset YAML file')
    train_parser.add_argument('--epochs', type=int, default=100,
                             help='Number of training epochs')
    train_parser.add_argument('--batch', type=int, default=16,
                             help='Batch size')
    train_parser.add_argument('--model', type=str, default='yolov8n',
                             help='Model size (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)')
    train_parser.add_argument('--weights', '-w', type=str,
                             help='Pretrained weights file')
    
    # Annotation command
    annotate_parser = subparsers.add_parser('annotate', help='Annotate images')
    annotate_parser.add_argument('--input', '-i', type=str, required=True,
                                help='Input image file or directory')
    
    # Data conversion command
    convert_parser = subparsers.add_parser('convert', help='Convert annotation formats')
    convert_parser.add_argument('--mode', '-m', choices=['xml2yolo', 'yolo2xml', 'validate'],
                               required=True, help='Conversion mode')
    convert_parser.add_argument('--input', '-i', type=str, required=True,
                               help='Input directory')
    convert_parser.add_argument('--output', '-o', type=str, required=True,
                               help='Output directory')
    convert_parser.add_argument('--images', type=str,
                               help='Images directory (required for yolo2xml)')
    
    # Detection command
    detect_parser = subparsers.add_parser('detect', help='Run object detection')
    detect_parser.add_argument('--input', '-i', type=str, required=True,
                              help='Input image or video file')
    detect_parser.add_argument('--output', '-o', type=str,
                              help='Output file')
    detect_parser.add_argument('--weights', '-w', type=str,
                              help='Model weights file')
    detect_parser.add_argument('--conf', type=float, default=0.5,
                              help='Confidence threshold')
    detect_parser.add_argument('--show', action='store_true',
                              help='Show results')
    detect_parser.add_argument('--benchmark', action='store_true',
                              help='Run benchmark test')
    
    # Prepare dataset command
    prepare_parser = subparsers.add_parser('prepare', help='Prepare dataset for training')
    prepare_parser.add_argument('--images', type=str, required=True,
                               help='Images directory')
    prepare_parser.add_argument('--annotations', type=str, required=True,
                               help='Annotations directory')
    prepare_parser.add_argument('--output', '-o', type=str, default='data/dataset',
                               help='Output dataset directory')
    prepare_parser.add_argument('--split', type=float, nargs=2, default=[0.8, 0.2],
                               help='Train/validation split ratio')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    try:
        if args.command == 'web':
            from src.frontend.app import TrafficCounterApp
            app = TrafficCounterApp()
            
            # Initialize models before launching the app
            logger.info("Attempting to initialize models on startup...")
            
            # First check for model files
            weights_file = config.get('paths.weights_file')
            default_model = config.get('model.name', 'yolov8n')
            
            if Path(weights_file).exists():
                logger.info(f"Found weights file at {weights_file}")
            elif Path(f"{default_model}.pt").exists():
                logger.info(f"Found default model at {default_model}.pt")
            else:
                logger.info(f"No model found at {weights_file} or {default_model}.pt")
                logger.info("The app will attempt to download a model if needed")
            
            app.launch(
                server_name=args.host,
                server_port=args.port,
                share=args.share,
                debug=args.debug
            )
        
        elif args.command == 'count':
            from src.detection.counter import main as count_main
            # Temporarily modify sys.argv for the counter main function
            original_argv = sys.argv[:]
            sys.argv = ['counter.py', '--input', args.input]
            if args.output:
                sys.argv.extend(['--output', args.output])
            if args.weights:
                sys.argv.extend(['--weights', args.weights])
            if args.line_y:
                sys.argv.extend(['--line-y', str(args.line_y)])
            if args.show:
                sys.argv.append('--show')
            
            count_main()
            sys.argv = original_argv
        
        elif args.command == 'train':
            from src.training.trainer import main as train_main
            original_argv = sys.argv[:]
            sys.argv = ['trainer.py', '--mode', 'train', '--data', args.data]
            if args.weights:
                sys.argv.extend(['--weights', args.weights])
            sys.argv.extend(['--model', args.model])
            
            # Update config for training parameters
            config.update_config('training.epochs', args.epochs)
            config.update_config('training.batch_size', args.batch)
            
            train_main()
            sys.argv = original_argv
        
        elif args.command == 'annotate':
            from src.annotation.annotator import main as annotate_main
            original_argv = sys.argv[:]
            sys.argv = ['annotator.py', '--input', args.input]
            
            annotate_main()
            sys.argv = original_argv
        
        elif args.command == 'convert':
            from src.annotation.data_converter import main as convert_main
            original_argv = sys.argv[:]
            sys.argv = ['data_converter.py', '--mode', args.mode, '--input', args.input, '--output', args.output]
            if args.images:
                sys.argv.extend(['--images', args.images])
            
            convert_main()
            sys.argv = original_argv
        
        elif args.command == 'detect':
            from src.detection.detector import main as detect_main
            original_argv = sys.argv[:]
            sys.argv = ['detector.py', '--input', args.input]
            if args.output:
                sys.argv.extend(['--output', args.output])
            if args.weights:
                sys.argv.extend(['--weights', args.weights])
            sys.argv.extend(['--conf', str(args.conf)])
            if args.show:
                sys.argv.append('--show')
            if args.benchmark:
                sys.argv.append('--benchmark')
            
            detect_main()
            sys.argv = original_argv
        
        elif args.command == 'prepare':
            from src.training.trainer import YOLOTrainer
            trainer = YOLOTrainer()
            dataset_yaml = trainer.prepare_dataset(
                args.images, 
                args.annotations, 
                tuple(args.split)
            )
            print(f"✅ Dataset prepared successfully!")
            print(f"Dataset YAML: {dataset_yaml}")
    
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error executing command '{args.command}': {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()