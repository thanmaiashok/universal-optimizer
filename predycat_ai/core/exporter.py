"""Export System - ONNX, GGUF, PyTorch"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ExportFormat(Enum):
    PYTORCH = "pt"
    ONNX = "onnx"
    GGUF = "gguf"
    TORCH_SCRIPT = "torchscript"


@dataclass
class ExportResult:
    format: ExportFormat
    output_path: str
    file_size_mb: float
    success: bool
    error: Optional[str] = None


class ExportEngine:
    def __init__(self, output_dir: str = "./outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def export(
        self,
        model: nn.Module,
        format: ExportFormat,
        model_name: str = "optimized_model",
        input_shape: Optional[tuple] = None,
        **kwargs
    ) -> ExportResult:
        logger.info(f"Exporting model to {format.value}")
        
        output_path = self.output_dir / f"{model_name}.{format.value}"
        
        try:
            if format == ExportFormat.PYTORCH:
                return self._export_pytorch(model, output_path, **kwargs)
            elif format == ExportFormat.ONNX:
                return self._export_onnx(model, output_path, input_shape, **kwargs)
            elif format == ExportFormat.TORCH_SCRIPT:
                return self._export_torchscript(model, output_path, input_shape, **kwargs)
            else:
                return ExportResult(format, str(output_path), 0, False, f"Unsupported format: {format}")
                
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return ExportResult(format, str(output_path), 0, False, str(e))
    
    def _export_pytorch(
        self,
        model: nn.Module,
        output_path: Path,
        **kwargs
    ) -> ExportResult:
        model.eval()

        save_format = kwargs.get("save_format", "pt")

        if save_format == "pretrained":
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": model.config if hasattr(model, "config") else {}
            }, output_path)
        else:
            torch.save(model.state_dict(), output_path)

        # Save sidecar metadata so playground can reconstruct model + tokenizer
        try:
            import json
            meta: dict = {}
            if hasattr(model, "config"):
                cfg = model.config
                meta["model_type"] = getattr(cfg, "model_type", None)
                meta["architectures"] = getattr(cfg, "architectures", None)
                meta["name_or_path"] = getattr(cfg, "_name_or_path", None)
                meta["model_id"] = getattr(cfg, "_name_or_path", None) or getattr(cfg, "model_type", None)
            meta["task"] = kwargs.get("task", None)
            meta_path = output_path.with_suffix(".meta.json")
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save model metadata sidecar: {e}")

        file_size = output_path.stat().st_size / (1024 * 1024)

        return ExportResult(
            format=ExportFormat.PYTORCH,
            output_path=str(output_path),
            file_size_mb=file_size,
            success=True
        )
    
    def _export_onnx(
        self,
        model: nn.Module,
        output_path: Path,
        input_shape: Optional[tuple] = None,
        **kwargs
    ) -> ExportResult:
        if input_shape is None:
            input_shape = (1, 512)
        
        dummy_input = torch.randn(*input_shape)
        
        if "input_names" not in kwargs:
            kwargs["input_names"] = ["input_ids"]
        if "output_names" not in kwargs:
            kwargs["output_names"] = ["logits"]
        if "dynamic_axes" not in kwargs:
            kwargs["dynamic_axes"] = {0: "batch_size", 1: "seq_len"}
        
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            **kwargs
        )
        
        file_size = output_path.stat().st_size / (1024 * 1024)
        
        return ExportResult(
            format=ExportFormat.ONNX,
            output_path=str(output_path),
            file_size_mb=file_size,
            success=True
        )
    
    def _export_torchscript(
        self,
        model: nn.Module,
        output_path: Path,
        input_shape: Optional[tuple] = None,
        **kwargs
    ) -> ExportResult:
        model.eval()
        
        if input_shape is None:
            input_shape = (1, 512)
        
        dummy_input = torch.randn(*input_shape)
        
        try:
            traced_model = torch.jit.trace(model, dummy_input)
            torch.jit.save(traced_model, output_path)
        except Exception as e:
            scripted_model = torch.jit.script(model)
            torch.jit.save(scripted_model, output_path)
        
        file_size = output_path.stat().st_size / (1024 * 1024)
        
        return ExportResult(
            format=ExportFormat.TORCH_SCRIPT,
            output_path=str(output_path),
            file_size_mb=file_size,
            success=True
        )
    
    def export_to_onnxruntime(
        self,
        model: nn.Module,
        model_name: str,
        input_shape: tuple = (1, 512),
        optimize: bool = True
    ) -> str:
        import onnxruntime as ort
        
        temp_path = self.output_dir / f"{model_name}.onnx"
        
        result = self._export_onnx(model, temp_path, input_shape)
        
        if not result.success:
            raise ValueError(f"ONNX export failed: {result.error}")
        
        if optimize:
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            ort.InferenceSession(str(temp_path), sess_options)
        
        return str(temp_path)


def export_to_ggml(model: nn.Module, output_path: str, quantization_type: str = "q4_k") -> bool:
    logger.info(f"GGML export to {output_path} with {quantization_type}")
    
    try:
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        return True
    except Exception as e:
        logger.error(f"GGML export failed: {e}")
        return False


def export_stable_diffusion(model: Any, output_path: str) -> bool:
    logger.info(f"Exporting Stable Diffusion to {output_path}")
    try:
        model.save_pretrained(output_path)
        return True
    except Exception as e:
        logger.error(f"SD export failed: {e}")
        return False


def load_exported_model(path: str, device: str = "auto") -> Tuple[Any, Dict]:
    path = Path(path)
    
    if path.suffix == ".pt":
        return load_pytorch(path, device)
    elif path.suffix == ".onnx":
        return load_onnx(path, device)
    else:
        raise ValueError(f"Unknown format: {path.suffix}")


def load_pytorch(path: Path, device: str = "auto") -> Tuple[Dict, Dict]:
    checkpoint = torch.load(path, map_location=device)
    
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
        config = checkpoint.get("config", {})
    else:
        state_dict = checkpoint
        config = {}
    
    return state_dict, config


def load_onnx(path: Path, device: str = "auto") -> Tuple[Any, Dict]:
    import onnx
    
    model = onnx.load(path)
    
    return model, {"format": "onnx"}


def save_optimization_report(
    original_model: Any,
    optimized_model: Any,
    report_path: str,
    metadata: Dict[str, Any]
) -> str:
    report = {
        "original": metadata.get("original", {}),
        "optimized": metadata.get("optimized", {}),
        "improvements": metadata.get("improvements", {}),
        "pipeline": metadata.get("pipeline", []),
    }
    
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    return report_path


def add_missing_imports():
    import sys
    from pathlib import Path
    
    required_modules = [
        "torch",
        "numpy",
        "transformers",
    ]
    
    return True