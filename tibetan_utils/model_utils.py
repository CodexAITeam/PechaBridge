"""
Model utilities for the TibetanOCR project.
"""

import os
from pathlib import Path
from typing import Dict, List, Union, Any, Optional
try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None
from ultralytics import YOLO


class ModelManager:
    """Model management utilities for YOLO models."""
    
    @staticmethod
    def load_model(model_path: str) -> YOLO:
        """
        Load a YOLO model.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            YOLO: Loaded model
        """
        return YOLO(model_path)
    
    @staticmethod
    def train_model(model: YOLO, data_path: str, epochs: int = 100, image_size: int = 1024,
                   batch_size: int = 16, workers: int = 8, device: str = '',
                   project: str = 'runs/detect', name: str = 'train',
                   patience: int = 50, use_wandb: bool = False, **kwargs) -> Dict[str, Any]:
        """
        Train a YOLO model.
        
        Args:
            model: YOLO model
            data_path: Path to the dataset configuration
            epochs: Number of training epochs
            image_size: Image size for training
            batch_size: Batch size for training
            workers: Number of workers for data loading
            device: Device for training
            project: Project name for output
            name: Experiment name
            patience: EarlyStopping patience in epochs
            use_wandb: Whether to use Weights & Biases logging
            **kwargs: Additional arguments for training
            
        Returns:
            Dict[str, Any]: Training results
        """
        # Prepare training arguments
        train_args = {
            'data': data_path,
            'epochs': epochs,
            'imgsz': image_size,
            'batch': batch_size,
            'workers': workers,
            'device': device,
            'project': project,
            'name': name,
            'patience': patience,
            'plots': True,
            'save_period': 10,
        }
        
        # Add wandb logging if enabled
        if use_wandb:
            train_args.update({
                'upload_dataset': True,
                'logger': 'wandb',
            })
        
        # Add additional arguments
        train_args.update(kwargs)
        
        # Start training
        results = model.train(**train_args)
        
        return results
    
    @staticmethod
    def export_model(model: YOLO, format: str = 'torchscript') -> str:
        """
        Export a YOLO model.
        
        Args:
            model: YOLO model
            format: Export format
            
        Returns:
            str: Path to the exported model
        """
        return model.export(format=format)
    
    @staticmethod
    def predict(model: YOLO, source: Union[str, List[str]], conf: float = 0.25,
               image_size: int = 1024, device: str = '', **kwargs) -> List[Any]:
        """
        Run inference with a YOLO model.
        
        Args:
            model: YOLO model
            source: Source for inference
            conf: Confidence threshold
            image_size: Image size for inference
            device: Device for inference
            **kwargs: Additional arguments for inference
            
        Returns:
            List[Any]: Inference results
        """
        # Prepare inference arguments
        predict_args = {
            'source': source,
            'conf': conf,
            'imgsz': image_size,
            'device': device,
        }
        
        # Add additional arguments
        predict_args.update(kwargs)
        
        # Run inference
        results = model.predict(**predict_args)
        
        return results
    
    @staticmethod
    def save_model_to_wandb(model_path: str, artifact_name: str,
                           artifact_type: str = 'model') -> Optional[Any]:
        """
        Save a model to Weights & Biases.
        
        Args:
            model_path: Path to the model file
            artifact_name: Name of the artifact
            artifact_type: Type of the artifact
            
        Returns:
            Optional[Any]: Artifact if wandb is initialized, None otherwise
        """
        if wandb is None:
            print("Warning: wandb is not installed. Skipping artifact logging.")
            return None

        if wandb.run is None:
            print("Warning: wandb.run is None. Make sure wandb is initialized.")
            return None
        
        artifact = wandb.Artifact(name=artifact_name, type=artifact_type)
        artifact.add_file(model_path)
        
        wandb.log_artifact(artifact)
        
        return artifact
    
    @staticmethod
    def get_best_model_path(project: str, name: str) -> str:
        """
        Get the path to the best model.
        
        Args:
            project: Project name
            name: Experiment name
            
        Returns:
            str: Path to the best model
        """
        return os.path.join(project, name, 'weights', 'best.pt')
