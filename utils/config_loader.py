"""
Configuration loader for YOLO Traffic Counter
"""
import yaml
import os
from pathlib import Path
from typing import Dict, Any


class ConfigLoader:
    """Configuration loader class"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
        with open(self.config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
            
        # Create directories if they don't exist
        self._create_directories(config)
        
        return config
    
    def _create_directories(self, config: Dict[str, Any]) -> None:
        """Create necessary directories"""
        directories = [
            config['data']['raw_dir'],
            config['data']['processed_dir'],
            config['data']['annotations_dir'],
            config['paths']['models_dir']
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key (supports nested keys with dot notation)"""
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data configuration"""
        return self.config.get('data', {})
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        return self.config.get('model', {})
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration"""
        return self.config.get('training', {})
    
    def get_detection_config(self) -> Dict[str, Any]:
        """Get detection configuration"""
        return self.config.get('detection', {})
    
    def get_counting_config(self) -> Dict[str, Any]:
        """Get counting configuration"""
        return self.config.get('counting', {})
    
    def get_classes(self) -> Dict[int, str]:
        """Get class mapping"""
        classes = self.config.get('classes', {})
        return {int(k): v for k, v in classes.items()}
    
    def get_paths(self) -> Dict[str, str]:
        """Get paths configuration"""
        return self.config.get('paths', {})
    
    def update_config(self, key: str, value: Any) -> None:
        """Update configuration value"""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save_config(self, path: str = None) -> None:
        """Save configuration to file"""
        save_path = Path(path) if path else self.config_path
        
        with open(save_path, 'w', encoding='utf-8') as file:
            yaml.dump(self.config, file, default_flow_style=False, allow_unicode=True)


# Global config instance
config = ConfigLoader()