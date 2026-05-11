"""Utility functions for PredycatAI"""

import logging
import os
import platform
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class HardwareInfo:
    cpu: str
    ram_gb: float
    has_cuda: bool
    cuda_version: Optional[str]
    has_mps: bool
    device_name: Optional[str]


def get_hardware_info() -> HardwareInfo:
    import torch
    
    cpu = platform.processor()
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    except:
        ram_gb = 0
    
    has_cuda = torch.cuda.is_available()
    cuda_version = None
    
    if has_cuda:
        try:
            cuda_version = torch.version.cuda
        except:
            pass
    
    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    
    device_name = None
    if has_cuda:
        try:
            device_name = torch.cuda.get_device_name(0)
        except:
            pass
    
    return HardwareInfo(
        cpu=cpu,
        ram_gb=ram_gb,
        has_cuda=has_cuda,
        cuda_version=cuda_version,
        has_mps=has_mps,
        device_name=device_name
    )


def check_dependencies() -> List[str]:
    required = [
        "torch",
        "numpy",
    ]
    
    optional = [
        "transformers",
        "bitsandbytes",
        "onnx",
        "onnxruntime",
        "fastapi",
        "uvicorn",
    ]
    
    available = []
    missing = []
    
    for mod in required + optional:
        try:
            __import__(mod)
            available.append(mod)
        except ImportError:
            missing.append(mod)
    
    return available, missing


def auto_device() -> str:
    import torch
    
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_memory_stats() -> Dict[str, float]:
    import torch
    
    stats = {}
    
    if torch.cuda.is_available():
        stats["vram_total_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        stats["vram_allocated_gb"] = torch.cuda.memory_allocated() / (1024 ** 3)
        stats["vram_reserved_gb"] = torch.cuda.memory_reserved() / (1024 ** 3)
        stats["vram_free_gb"] = stats["vram_total_gb"] - stats["vram_allocated_gb"]
    else:
        stats["vram_total_gb"] = 0
        stats["vram_allocated_gb"] = 0
    
    try:
        import psutil
        virt = psutil.virtual_memory()
        stats["ram_total_gb"] = virt.total / (1024 ** 3)
        stats["ram_available_gb"] = virt.available / (1024 ** 3)
        stats["ram_percent"] = virt.percent
    except:
        stats["ram_total_gb"] = 0
        stats["ram_available_gb"] = 0
        stats["ram_percent"] = 0
    
    return stats


def format_size(size_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} PB"


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    return f"{seconds/3600:.1f}h"


def setup_output_dir(path: str = "./outputs") -> str:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    (output / "models").mkdir(exist_ok=True)
    (output / "reports").mkdir(exist_ok=True)
    (output / "logs").mkdir(exist_ok=True)
    return str(output)


def create_sample_model() -> Any:
    import torch.nn as nn
    
    class SimpleModel(nn.Module):
        def __init__(self, vocab_size=32000, hidden_size=768, num_layers=12):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, hidden_size)
            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=12,
                    dim_feedforward=hidden_size * 4,
                    batch_first=True
                ) for _ in range(num_layers)
            ])
            self.output = nn.Linear(hidden_size, vocab_size)
            
        def forward(self, x):
            x = self.embedding(x)
            for layer in self.layers:
                x = layer(x)
            return self.output(x)
    
    return SimpleModel()


def download_sample_model(model_id: str = "gpt2") -> Any:
    try:
        from transformers import AutoModel, AutoTokenizer
        
        model = AutoModel.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        return model, tokenizer
    except Exception as e:
        logger.warning(f"Failed to download {model_id}: {e}")
        return create_sample_model(), None