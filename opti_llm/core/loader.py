"""Universal Model Loader - Supports HuggingFace, PyTorch, ONNX"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn
import numpy as np

logger = logging.getLogger(__name__)


class ModelSource(Enum):
    HUGGINGFACE = "huggingface"
    PYTORCH = "pytorch"
    ONNX = "onnx"
    UNKNOWN = "unknown"


class ModelArchitecture(Enum):
    LLM = "llm"
    TRANSFORMER = "transformer"
    CNN = "cnn"
    RNN = "rnn"
    HYBRID = "hybrid"
    UNKNOWN = "unknown"


@dataclass
class ModelInfo:
    source: ModelSource
    architecture: ModelArchitecture
    num_parameters: int
    num_layers: int
    hidden_size: Optional[int] = None
    num_heads: Optional[int] = None
    intermediate_size: Optional[int] = None
    vocab_size: Optional[int] = None
    layer_types: Optional[List[str]] = None
    model_name: Optional[str] = None
    config: Optional[Dict[str, Any]] = None


class UniversalLoader:
    def __init__(self, device: str = "auto"):
        self.device = self._get_device(device)
        self.cache = {}
        
    def _get_device(self, device: str) -> str:
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return device
    
    def detect_source(self, path: str) -> ModelSource:
        _FILE_EXTS = {".pt", ".pth", ".bin", ".onnx", ".safetensors", ".gguf", ".ggml"}
        p = Path(path)
        if p.suffix in {".pt", ".pth", ".bin", ".safetensors", ".gguf", ".ggml"}:
            return ModelSource.PYTORCH
        if p.suffix == ".onnx":
            return ModelSource.ONNX
        if p.is_dir() and (p / "config.json").exists():
            return ModelSource.HUGGINGFACE
        # HuggingFace slug: no real file extension (version numbers like 3.2-1B don't count)
        if p.suffix not in _FILE_EXTS and not p.is_absolute():
            return ModelSource.HUGGINGFACE
        return ModelSource.UNKNOWN
    
    def detect_architecture(self, model_config: Dict) -> ModelArchitecture:
        model_type = model_config.get("model_type", "").lower()
        
        if any(t in model_type for t in ["gpt", "llama", "mistral", "falcon", "qwen", "bloom"]):
            return ModelArchitecture.LLM
        elif any(t in model_type for t in ["bert", "roberta", "t5", "bart", "encoder"]):
            return ModelArchitecture.TRANSFORMER
        elif any(t in model_type for t in ["resnet", "vit", "conv", "efficientnet"]):
            return ModelArchitecture.CNN
        elif "rnn" in model_type or "lstm" in model_type:
            return ModelArchitecture.RNN
        elif any(t in model_type for t in ["llava", "vision"]):
            return ModelArchitecture.HYBRID
        
        vocab_size = model_config.get("vocab_size", 0)
        hidden_size = model_config.get("hidden_size", 0)
        
        if vocab_size > 50000 and hidden_size > 768:
            return ModelArchitecture.LLM
        elif hidden_size > 0:
            return ModelArchitecture.TRANSFORMER
            
        return ModelArchitecture.UNKNOWN
    
    def load_model(self, model_path: str, **kwargs) -> Tuple[Any, ModelInfo]:
        source = self.detect_source(model_path)
        logger.info(f"Detected source: {source.value}")
        
        if source == ModelSource.HUGGINGFACE:
            return self._load_huggingface(model_path, **kwargs)
        elif source == ModelSource.PYTORCH:
            return self._load_pytorch(model_path, **kwargs)
        elif source == ModelSource.ONNX:
            return self._load_onnx(model_path, **kwargs)
        else:
            raise ValueError(f"Unknown model source: {model_path}")
    
    def _load_huggingface(self, model_path: str, **kwargs):
        from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoConfig

        cache_dir = kwargs.get("cache_dir", "./models")
        trust_remote = kwargs.get("trust_remote_code", True)

        config = AutoConfig.from_pretrained(model_path, cache_dir=cache_dir, trust_remote_code=trust_remote)
        model_type = getattr(config, "model_type", "").lower()

        # pick the right class so we get a loss head for perplexity measurement
        causal_types = {"gpt", "llama", "mistral", "falcon", "qwen", "bloom", "opt", "gemma", "phi"}
        seq2seq_types = {"t5", "bart", "pegasus", "marian"}

        try:
            if any(t in model_type for t in causal_types):
                model = AutoModelForCausalLM.from_pretrained(model_path, cache_dir=cache_dir, trust_remote_code=trust_remote)
            elif any(t in model_type for t in seq2seq_types):
                model = AutoModelForSeq2SeqLM.from_pretrained(model_path, cache_dir=cache_dir, trust_remote_code=trust_remote)
            else:
                model = AutoModel.from_pretrained(model_path, cache_dir=cache_dir, trust_remote_code=trust_remote)
        except Exception as e:
            logger.warning(f"Specialised load failed ({e}), falling back to AutoModel")
            model = AutoModel.from_pretrained(model_path, cache_dir=cache_dir, trust_remote_code=trust_remote)

        model = model.to(self.device)
        model.eval()
        
        info = self._extract_model_info(model, config)
        info.source = ModelSource.HUGGINGFACE
        info.model_name = model_path.split("/")[-1]
        
        return model, info
    
    def _load_pytorch(self, model_path: str, **kwargs):
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        num_params = sum(p.numel() for p in state_dict.values() if isinstance(p, torch.Tensor))
        
        class DummyConfig:
            model_type = "custom"
            vocab_size = kwargs.get("vocab_size", 32000)
            hidden_size = kwargs.get("hidden_size", 4096)
            num_hidden_layers = kwargs.get("num_layers", 32)
            num_attention_heads = kwargs.get("num_heads", 32)
            
        info = ModelInfo(
            source=ModelSource.PYTORCH,
            architecture=ModelArchitecture.UNKNOWN,
            num_parameters=num_params,
            num_layers=DummyConfig.num_hidden_layers,
            hidden_size=DummyConfig.hidden_size,
            num_heads=DummyConfig.num_attention_heads,
            vocab_size=DummyConfig.vocab_size
        )
        
        return state_dict, info
    
    def _load_onnx(self, model_path: str, **kwargs):
        import onnx
        
        model = onnx.load(model_path)
        graph = model.graph
        
        num_params = 0
        for initializer in graph.initializer:
            shape = [d for d in initializer.dims]
            num_params += np.prod(shape) if shape else 0
        
        info = ModelInfo(
            source=ModelSource.ONNX,
            architecture=ModelArchitecture.UNKNOWN,
            num_parameters=num_params,
            num_layers=len(graph.node)
        )
        
        return model, info
    
    def _extract_model_info(self, model, config) -> ModelInfo:
        architecture = self.detect_architecture(config.to_dict() if hasattr(config, "to_dict") else {})
        
        num_params = sum(p.numel() for p in model.parameters())
        num_layers = getattr(config, "num_hidden_layers", 0) or getattr(config, "num_layers", 0) or 0
        
        return ModelInfo(
            source=ModelSource.HUGGINGFACE,
            architecture=architecture,
            num_parameters=num_params,
            num_layers=num_layers,
            hidden_size=getattr(config, "hidden_size", None),
            num_heads=getattr(config, "num_attention_heads", None),
            intermediate_size=getattr(config, "intermediate_size", None),
            vocab_size=getattr(config, "vocab_size", None),
            config=config.to_dict() if hasattr(config, "to_dict") else {}
        )


def auto_detect_model_type(model_or_config: Any) -> str:
    if hasattr(model_or_config, "model_type"):
        model_type = model_or_config.model_type.lower()
        if "gpt" in model_type or "llama" in model_type:
            return "llm"
        elif "bert" in model_type or "roberta" in model_type:
            return "transformer"
        elif "resnet" in model_type or "vit" in model_type:
            return "cnn"
    return "unknown"


def get_model_memory_estimate(model: Any, dtype: Optional[torch.dtype] = None) -> int:
    total = 0
    for param in model.parameters() if hasattr(model, "parameters") else []:
        if isinstance(param, torch.Tensor):
            size = param.numel() * param.element_size()
            total += size
    return total


def get_model_size_mb(model_path: str) -> float:
    path = Path(model_path)
    if path.is_file():
        return path.stat().st_size / (1024 * 1024)
    elif path.is_dir():
        total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        return total / (1024 * 1024)
    return 0.0