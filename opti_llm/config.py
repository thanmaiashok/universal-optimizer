"""OptiLLM Configuration System"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from datetime import datetime


@dataclass
class OptiLLMConfig:
    version: str = "1.0.0"
    target_platform: str = "laptop"
    preset: str = "balanced"
    max_accuracy_drop: float = 2.0
    max_size_mb: Optional[float] = None
    max_ram_mb: Optional[float] = None
    export_formats: List[str] = field(default_factory=lambda: ["pt"])
    use_quantization: bool = True
    use_pruning: bool = True
    use_distillation: bool = True
    use_finetuning: bool = False
    enable_reasoning_check: bool = True
    output_dir: str = "./outputs"
    cache_dir: str = "./models"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OptiLLMConfig":
        return cls(**data)
    
    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "OptiLLMConfig":
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def validate(self) -> List[str]:
        errors = []
        
        if self.target_platform not in ["laptop", "cloud", "mobile", "edge"]:
            errors.append(f"Invalid platform: {self.target_platform}")
        
        if self.preset not in ["balanced", "ultra_compression", "max_accuracy", "mobile_safe"]:
            errors.append(f"Invalid preset: {self.preset}")
        
        if self.max_accuracy_drop < 0 or self.max_accuracy_drop > 10:
            errors.append("max_accuracy_drop must be between 0 and 10")
        
        return errors


def get_default_config() -> OptiLLMConfig:
    return OptiLLMConfig()


def get_config_for_platform(platform: str) -> OptiLLMConfig:
    configs = {
        "laptop": OptiLLMConfig(
            target_platform="laptop",
            preset="balanced",
            max_accuracy_drop=2.0,
            export_formats=["pt", "onnx"]
        ),
        "cloud": OptiLLMConfig(
            target_platform="cloud",
            preset="max_accuracy",
            max_accuracy_drop=1.0,
            export_formats=["pt", "onnx"]
        ),
        "mobile": OptiLLMConfig(
            target_platform="mobile",
            preset="balanced",
            max_accuracy_drop=2.0,
            max_size_mb=300,
            max_ram_mb=2048,
            export_formats=["pt", "onnx"]
        ),
        "edge": OptiLLMConfig(
            target_platform="edge",
            preset="ultra_compression",
            max_accuracy_drop=3.0,
            max_size_mb=100,
            max_ram_mb=1024,
            export_formats=["onnx"]
        ),
    }
    
    return configs.get(platform.lower(), OptiLLMConfig())


def create_default_dirs(base_dir: str = "."):
    dirs = [
        "outputs",
        "outputs/models",
        "outputs/reports",
        "outputs/logs",
        "models",
        "cache",
    ]
    
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


_global_config: Optional[OptiLLMConfig] = None


def get_global_config() -> OptiLLMConfig:
    global _global_config
    if _global_config is None:
        _global_config = get_default_config()
    return _global_config


def set_global_config(config: OptiLLMConfig):
    global _global_config
    _global_config = config