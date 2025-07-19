"""
Gradio-based frontend for YOLO Traffic Counter
"""
import gradio as gr
import cv2
import numpy as np
import json
import tempfile
import os
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime
import threading
import time
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.detection.detector import YOLODetector
from src.detection.counter import TrafficCounter
from src.training.trainer import YOLOTrainer
from src.annotation.annotator import ImageAnnotator
from src.annotation.data_converter import DataConverter
from utils.config_loader import config
from utils.logger import logger


class TrafficCounterApp:
    """Main application class for traffic counter frontend"""
    
    def __init__(self):
        self.detector = None
        self.counter = None
        self.trainer = None
        self.annotator = ImageAnnotator()
        self.converter = DataConverter()
        
        # State
        self.current_video_path = None
        self.processing_stats = {}
        self.is_processing = False
        
        logger.info("Traffic Counter App initialized")
    
    def initialize_models(self, weights_path: str = None) -> str:
        """Initialize YOLO models"""
        try:
            self.detector = YOLODetector(weights_path)
            self.counter = TrafficCounter(self.detector)
            self.trainer = YOLOTrainer()
            
            model_info = self.detector.get_model_info()
            
            return f"✅ Models initialized successfully!\n\nModel Info:\n{json.dumps(model_info, indent=2)}"
        
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            return f"❌ Error initializing models: {str(e)}"
    
    def process_video_for_counting(self, video_file, confidence_threshold: float = 0.5,
                                 line_position: float = 0.5, show_tracks: bool = True) -> Tuple[str, str, str]:
        """Process video for traffic counting"""
        if video_file is None:
            return "❌ Please upload a video file", "", ""
        
        if self.counter is None:
            return "❌ Please initialize models first", "", ""
        
        try:
            self.is_processing = True
            
            # Update detection threshold
            self.detector.update_thresholds(confidence=confidence_threshold)
            
            # Create temporary output file
            output_path = tempfile.mktemp(suffix='.mp4')
            
            # Reset counter
            self.counter.reset_counts()
            
            # Get video dimensions for counting line
            cap = cv2.VideoCapture(video_file.name)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            # Add counting line
            line_y = int(height * line_position)
            self.counter.counting_lines.clear()  # Clear existing lines
            self.counter.add_horizontal_counting_line(line_y, width, "counting_line")
            
            # Process video
            stats = self.counter.process_video(
                video_file.name,
                output_path=output_path,
                show_video=False,
                save_stats=True
            )
            
            self.processing_stats = stats
            self.current_video_path = output_path
            
            # Create results summary
            results_text = self._format_counting_results(stats)
            
            # Create visualization
            viz_html = self._create_counting_visualization(stats)
            
            self.is_processing = False
            
            return results_text, output_path, viz_html
        
        except Exception as e:
            self.is_processing = False
            logger.error(f"Error processing video: {e}")
            return f"❌ Error processing video: {str(e)}", "", ""
    
    def _format_counting_results(self, stats: Dict[str, Any]) -> str:
        """Format counting results for display"""
        results = []
        results.append("🎯 **Traffic Counting Results**\n")
        
        # Final counts
        final_counts = stats.get('final_counts', {})
        total_count = sum(final_counts.values())
        
        results.append("**📊 Final Counts:**")
        for class_name, count in final_counts.items():
            results.append(f"  • {class_name.title()}: **{count}**")
        results.append(f"  • **Total: {total_count}**\n")
        
        # Processing stats
        results.append("**⚡ Processing Statistics:**")
        results.append(f"  • Processed Frames: {stats.get('processed_frames', 0)}")
        results.append(f"  • Processing Time: {stats.get('processing_time', 0):.2f}s")
        results.append(f"  • Processing FPS: {stats.get('fps', 0):.2f}")
        
        return "\n".join(results)
    
    def _create_counting_visualization(self, stats: Dict[str, Any]) -> str:
        """Create visualization charts for counting results"""
        try:
            # Create counts over time chart
            frame_stats = stats.get('frame_stats', [])
            if not frame_stats:
                return "<p>No frame statistics available</p>"
            
            df = pd.DataFrame(frame_stats)
            
            # Counts over time
            fig_timeline = go.Figure()
            
            final_counts = stats.get('final_counts', {})
            for class_name in final_counts.keys():
                class_counts = []
                running_count = 0
                
                for frame_stat in frame_stats:
                    counts = frame_stat.get('counts', {})
                    running_count = counts.get(class_name, running_count)
                    class_counts.append(running_count)
                
                fig_timeline.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=class_counts,
                    mode='lines',
                    name=class_name.title(),
                    line=dict(width=3)
                ))
            
            fig_timeline.update_layout(
                title="Traffic Count Over Time",
                xaxis_title="Time (seconds)",
                yaxis_title="Cumulative Count",
                template="plotly_white",
                height=400
            )
            
            # Final counts pie chart
            fig_pie = px.pie(
                values=list(final_counts.values()),
                names=[name.title() for name in final_counts.keys()],
                title="Final Count Distribution"
            )
            fig_pie.update_layout(height=400)
            
            # Convert to HTML
            timeline_html = fig_timeline.to_html(include_plotlyjs='cdn', div_id="timeline_chart")
            pie_html = fig_pie.to_html(include_plotlyjs='cdn', div_id="pie_chart")
            
            return f"""
            <div style="display: flex; flex-direction: column; gap: 20px;">
                <div>{timeline_html}</div>
                <div>{pie_html}</div>
            </div>
            """
        
        except Exception as e:
            logger.error(f"Error creating visualization: {e}")
            return f"<p>Error creating visualization: {str(e)}</p>"
    
    def train_model(self, images_folder, annotations_folder, epochs: int = 50,
                   batch_size: int = 16, model_name: str = "yolov8n") -> str:
        """Train YOLO model"""
        if not images_folder or not annotations_folder:
            return "❌ Please provide both images and annotations folders"
        
        try:
            # Initialize trainer
            self.trainer = YOLOTrainer(model_name)
            
            # Prepare dataset
            dataset_yaml = self.trainer.prepare_dataset(images_folder, annotations_folder)
            
            # Update training config
            config.update_config('training.epochs', epochs)
            config.update_config('training.batch_size', batch_size)
            
            # Train model
            results = self.trainer.train(dataset_yaml)
            
            return f"✅ Training completed successfully!\n\nBest model saved to: models/best.pt"
        
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return f"❌ Error training model: {str(e)}"
    
    def convert_annotations(self, annotations_folder, images_folder, mode: str = "xml2yolo") -> str:
        """Convert annotation formats"""
        if not annotations_folder:
            return "❌ Please provide annotations folder"
        
        try:
            output_folder = f"{annotations_folder}_converted"
            
            if mode == "xml2yolo":
                self.converter.convert_xml_directory_to_yolo(
                    annotations_folder, output_folder, images_folder
                )
                return f"✅ XML annotations converted to YOLO format!\n\nOutput saved to: {output_folder}"
            
            elif mode == "yolo2xml":
                if not images_folder:
                    return "❌ Images folder required for YOLO to XML conversion"
                
                self.converter.convert_yolo_directory_to_xml(
                    annotations_folder, images_folder, output_folder
                )
                return f"✅ YOLO annotations converted to XML format!\n\nOutput saved to: {output_folder}"
            
            else:
                return "❌ Invalid conversion mode"
        
        except Exception as e:
            logger.error(f"Error converting annotations: {e}")
            return f"❌ Error converting annotations: {str(e)}"
    
    def validate_annotations(self, annotations_folder) -> str:
        """Validate annotation files"""
        if not annotations_folder:
            return "❌ Please provide annotations folder"
        
        try:
            results = self.converter.validate_annotations(annotations_folder)
            
            validation_text = []
            validation_text.append("📋 **Annotation Validation Results**\n")
            validation_text.append(f"**Total Files:** {results['total_files']}")
            validation_text.append(f"**Valid Files:** {results['valid_files']} ✅")
            validation_text.append(f"**Invalid Files:** {results['invalid_files']} ❌")
            
            if results['errors']:
                validation_text.append(f"\n**Errors Found:** {len(results['errors'])}")
                validation_text.append("\n**First 10 Errors:**")
                for error in results['errors'][:10]:
                    validation_text.append(f"  • {error}")
                
                if len(results['errors']) > 10:
                    validation_text.append(f"  ... and {len(results['errors']) - 10} more errors")
            
            return "\n".join(validation_text)
        
        except Exception as e:
            logger.error(f"Error validating annotations: {e}")
            return f"❌ Error validating annotations: {str(e)}"
    
    def get_model_info(self) -> str:
        """Get current model information"""
        if self.detector is None:
            return "❌ No model loaded"
        
        try:
            info = self.detector.get_model_info()
            return f"**Model Information:**\n\n```json\n{json.dumps(info, indent=2)}\n```"
        
        except Exception as e:
            return f"❌ Error getting model info: {str(e)}"
    
    def create_interface(self) -> gr.Interface:
        """Create Gradio interface"""
        
        # Custom CSS
        css = """
        .gradio-container {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .tab-nav {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        }
        .progress-bar {
            background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        }
        """
        
        with gr.Blocks(css=css, title="YOLO Traffic Counter") as interface:
            gr.Markdown("""
            # 🚗 YOLO Traffic Counter
            
            A comprehensive traffic analysis system using YOLO object detection.
            Upload videos to count vehicles and pedestrians with real-time visualization.
            """)
            
            with gr.Tabs():
                # Tab 1: Traffic Counting
                with gr.TabItem("🎯 Traffic Counting"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### 📤 Upload & Settings")
                            
                            video_input = gr.File(
                                label="Upload Video",
                                file_types=[".mp4", ".avi", ".mov", ".mkv"],
                                type="file"
                            )
                            
                            with gr.Row():
                                confidence_slider = gr.Slider(
                                    minimum=0.1,
                                    maximum=1.0,
                                    value=0.5,
                                    step=0.1,
                                    label="Confidence Threshold"
                                )
                                
                                line_position_slider = gr.Slider(
                                    minimum=0.1,
                                    maximum=0.9,
                                    value=0.5,
                                    step=0.1,
                                    label="Counting Line Position"
                                )
                            
                            show_tracks_checkbox = gr.Checkbox(
                                value=True,
                                label="Show Object Tracks"
                            )
                            
                            process_btn = gr.Button(
                                "🚀 Process Video",
                                variant="primary",
                                size="lg"
                            )
                        
                        with gr.Column(scale=2):
                            gr.Markdown("### 📊 Results")
                            
                            results_text = gr.Markdown()
                            
                            output_video = gr.Video(
                                label="Processed Video",
                                height=400
                            )
                            
                            visualization = gr.HTML(label="Analytics")
                    
                    process_btn.click(
                        fn=self.process_video_for_counting,
                        inputs=[video_input, confidence_slider, line_position_slider, show_tracks_checkbox],
                        outputs=[results_text, output_video, visualization]
                    )
                
                # Tab 2: Model Training
                with gr.TabItem("🎓 Model Training"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### 📁 Dataset Configuration")
                            
                            images_folder = gr.Textbox(
                                label="Images Folder Path",
                                placeholder="/path/to/images"
                            )
                            
                            annotations_folder = gr.Textbox(
                                label="Annotations Folder Path", 
                                placeholder="/path/to/annotations"
                            )
                            
                            with gr.Row():
                                epochs_input = gr.Number(
                                    value=50,
                                    label="Training Epochs",
                                    minimum=1,
                                    maximum=500
                                )
                                
                                batch_size_input = gr.Number(
                                    value=16,
                                    label="Batch Size",
                                    minimum=1,
                                    maximum=64
                                )
                            
                            model_name_dropdown = gr.Dropdown(
                                choices=["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"],
                                value="yolov8n",
                                label="Model Size"
                            )
                            
                            train_btn = gr.Button(
                                "🎯 Start Training",
                                variant="primary"
                            )
                        
                        with gr.Column():
                            gr.Markdown("### 📈 Training Progress")
                            
                            training_output = gr.Markdown()
                    
                    train_btn.click(
                        fn=self.train_model,
                        inputs=[images_folder, annotations_folder, epochs_input, batch_size_input, model_name_dropdown],
                        outputs=[training_output]
                    )
                
                # Tab 3: Data Management
                with gr.TabItem("📝 Data Management"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### 🔄 Format Conversion")
                            
                            conv_annotations_folder = gr.Textbox(
                                label="Annotations Folder",
                                placeholder="/path/to/annotations"
                            )
                            
                            conv_images_folder = gr.Textbox(
                                label="Images Folder (for YOLO→XML)",
                                placeholder="/path/to/images"
                            )
                            
                            conversion_mode = gr.Radio(
                                choices=["xml2yolo", "yolo2xml"],
                                value="xml2yolo",
                                label="Conversion Mode"
                            )
                            
                            convert_btn = gr.Button(
                                "🔄 Convert Annotations",
                                variant="primary"
                            )
                            
                            conversion_output = gr.Markdown()
                        
                        with gr.Column():
                            gr.Markdown("### ✅ Validation")
                            
                            val_annotations_folder = gr.Textbox(
                                label="Annotations Folder",
                                placeholder="/path/to/annotations"
                            )
                            
                            validate_btn = gr.Button(
                                "✅ Validate Annotations",
                                variant="secondary"
                            )
                            
                            validation_output = gr.Markdown()
                    
                    convert_btn.click(
                        fn=self.convert_annotations,
                        inputs=[conv_annotations_folder, conv_images_folder, conversion_mode],
                        outputs=[conversion_output]
                    )
                    
                    validate_btn.click(
                        fn=self.validate_annotations,
                        inputs=[val_annotations_folder],
                        outputs=[validation_output]
                    )
                
                # Tab 4: Model Management
                with gr.TabItem("🔧 Model Management"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### 🎯 Model Initialization")
                            
                            weights_path_input = gr.Textbox(
                                label="Custom Weights Path (Optional)",
                                placeholder="/path/to/weights.pt"
                            )
                            
                            init_btn = gr.Button(
                                "🚀 Initialize Models",
                                variant="primary"
                            )
                            
                            init_output = gr.Markdown()
                        
                        with gr.Column():
                            gr.Markdown("### ℹ️ Model Information")
                            
                            info_btn = gr.Button(
                                "📊 Get Model Info",
                                variant="secondary"
                            )
                            
                            info_output = gr.Markdown()
                    
                    init_btn.click(
                        fn=self.initialize_models,
                        inputs=[weights_path_input],
                        outputs=[init_output]
                    )
                    
                    info_btn.click(
                        fn=self.get_model_info,
                        outputs=[info_output]
                    )
            
            # Footer
            gr.Markdown("""
            ---
            
            ### 📚 How to Use:
            
            1. **Traffic Counting**: Upload a video and adjust settings to count vehicles and pedestrians
            2. **Model Training**: Provide labeled data to train custom YOLO models  
            3. **Data Management**: Convert between annotation formats and validate data quality
            4. **Model Management**: Initialize models and view configuration details
            
            **Supported formats**: MP4, AVI, MOV, MKV for videos | JPG, PNG for images | XML (Pascal VOC) and TXT (YOLO) for annotations
            """)
        
        return interface
    
    def launch(self, **kwargs):
        """Launch the Gradio interface"""
        interface = self.create_interface()
        
        # Default launch parameters
        launch_params = {
            'server_name': config.get('frontend.host', 'localhost'),
            'server_port': config.get('frontend.port', 8501),
            'share': False,
            'debug': False
        }
        
        # Update with any provided parameters
        launch_params.update(kwargs)
        
        logger.info(f"Launching Traffic Counter App on {launch_params['server_name']}:{launch_params['server_port']}")
        
        return interface.launch(**launch_params)


def main():
    """Main function to launch the application"""
    import argparse
    
    parser = argparse.ArgumentParser(description="YOLO Traffic Counter Web App")
    parser.add_argument("--host", type=str, default="localhost", help="Host address")
    parser.add_argument("--port", type=int, default=8501, help="Port number")
    parser.add_argument("--share", action="store_true", help="Create public link")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Create and launch app
    app = TrafficCounterApp()
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        debug=args.debug
    )


if __name__ == "__main__":
    main()