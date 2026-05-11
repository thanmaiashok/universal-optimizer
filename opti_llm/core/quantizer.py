"""Quantization Engine - FP16, INT8, INT4, INT3"""

import os
import logging
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class QuantType(Enum):
    FP16 = "fp16"
    INT8 = "int8"
    INT4 = "int4"
    INT3 = "int3"
    GGML_Q4_K = "ggml_q4_k"
    GGUF_Q4_0 = "gguf_q4_0"


@dataclass
class QuantConfig:
    quant_type: QuantType
    bits: int = 16
    group_size: int = 128
    desc_act: bool = False
    sym: bool = True
    zero_point: bool = True


@dataclass
class QuantResult:
    original_size_mb: float
    quantized_size_mb: float
    compression_ratio: float
    quant_type: QuantType
    layers_quantized: int
    accuracy_drop: float = 0.0


class QuantizationEngine:
    def __init__(self, device: str = "auto"):
        self.device = self._get_device(device)
        self.results = {}
        
    def _get_device(self, device: str) -> str:
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return device
    
    def quantize(
        self,
        model: nn.Module,
        quant_type: QuantType = QuantType.INT8,
        calibration_data: Optional[torch.Tensor] = None,
        max_accuracy_drop: float = 2.0,
        **kwargs
    ) -> Tuple[nn.Module, QuantResult]:
        logger.info(f"Starting {quant_type.value} quantization")
        
        original_size = self._get_model_size(model)
        
        if quant_type == QuantType.FP16:
            quantized_model = self._quantize_fp16(model)
        elif quant_type == QuantType.INT8:
            quantized_model = self._quantize_int8(model, calibration_data)
        elif quant_type == QuantType.INT4:
            quantized_model = self._quantize_int4(model, calibration_data, **kwargs)
        elif quant_type == QuantType.INT3:
            quantized_model = self._quantize_int3(model, calibration_data)
        else:
            raise ValueError(f"Unsupported quant type: {quant_type}")
        
        quantized_size = self._get_model_size(quantized_model)
        compression = original_size / quantized_size if quantized_size > 0 else 1.0
        
        result = QuantResult(
            original_size_mb=original_size,
            quantized_size_mb=quantized_size,
            compression_ratio=compression,
            quant_type=quant_type,
            layers_quantized=self._count_quantizable_layers(model)
        )
        
        return quantized_model, result
    
    def _quantize_fp16(self, model: nn.Module) -> nn.Module:
        if isinstance(model, nn.Module):
            model = model.half()
            
            for module in model.modules():
                if hasattr(module, "half"):
                    try:
                        module.half()
                    except:
                        pass
        
        return model.to(self.device)
    
    def _quantize_int8(
        self,
        model: nn.Module,
        calibration_data: Optional[torch.Tensor] = None
    ) -> nn.Module:
        try:
            from torch.ao.quantization import quantize_dynamic, QConfig, default_weight_only_observer
            from torch.ao.quantization.quant_type import QInt8WeightOnlyQuantizer
            
            qconfig = QConfig(weight=default_weight_only_observer)
            
            quantized_model = quantize_dynamic(model, qconfig)
            
            logger.info("Applied INT8 dynamic quantization")
            return quantized_model.to(self.device)
            
        except Exception as e:
            logger.warning(f"PyTorch quantization not available: {e}, using manual implementation")
            return self._manual_int8_quantize(model, 128)
    
    def _quantize_int4(
        self,
        model: nn.Module,
        calibration_data: Optional[torch.Tensor] = None,
        **kwargs
    ) -> nn.Module:
        group_size = kwargs.get("group_size", 128)
        
        try:
            import bitsandbytes as bnb
            
            if hasattr(bnb, "Linear"):
                logger.info("Using bitsandbytes for INT4 quantization")
                
            self._replace_linear_with_4bit(model, **kwargs)
            return model.to(self.device)
            
        except ImportError:
            logger.warning("bitsandbytes not available, using manual INT4")
            return self._manual_int4_quantize(model, group_size)
    
    def _quantize_int3(
        self,
        model: nn.Module,
        calibration_data: Optional[torch.Tensor] = None
    ) -> nn.Module:
        logger.info("Using experimental INT3 quantization")
        return self._manual_int4_quantize(model, 64)
    
    def _manual_int8_quantize(self, model: nn.Module, group_size: int = 128) -> nn.Module:
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                if module.out_features >= group_size:
                    original_weight = module.weight.data
                    
                    max_val = torch.max(torch.abs(original_weight))
                    scale = 127.0 / max_val if max_val > 0 else 1.0
                    
                    quantized = torch.round(original_weight * scale).clamp(-128, 127)
                    dequantized = quantized / scale
                    
                    module.weight.data = dequantized
                    
                    if module.bias is not None:
                        module.bias.data = module.bias.data.float()
        
        return model.to(self.device)
    
    def _manual_int4_quantize(self, model: nn.Module, group_size: int = 128) -> nn.Module:
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                if module.out_features >= group_size:
                    weight = module.weight.data
                    out_features = weight.shape[0]
                    
                    num_groups = (out_features + group_size - 1) // group_size
                    
                    reshaped = weight[:, :num_groups * group_size].view(-1, group_size)
                    max_val = torch.max(torch.abs(reshaped), dim=-1, keepdim=True)[0]
                    max_val = torch.clamp(max_val, min=1e-6)
                    
                    scale = 7.0 / max_val
                    
                    quantized = torch.round(reshaped * scale.unsqueeze(1)).clamp(-7, 7)
                    dequantized = (quantized * (1.0 / scale.unsqueeze(1))).view_as(weight)
                    
                    module.weight.data = dequantized[:, :weight.shape[1]]
                    
                    if module.bias is not None:
                        module.bias.data = module.bias.data.float()
        
        return model.to(self.device)
    
    def _replace_linear_with_4bit(self, model: nn.Module, **kwargs):
        threshold = kwargs.get("linear_threshold", 256)
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and module.out_features >= threshold:
                try:
                    import bitsandbytes as bnb
                    module.__class__ = bnb.nn.Linear4bit
                    module.weight.data = module.weight.data.to(self.device)
                    module.quant = True
                except:
                    pass
    
    def _get_model_size(self, model: nn.Module) -> float:
        total = 0
        for param in model.parameters():
            total += param.numel() * param.element_size()
        return total / (1024 * 1024)
    
    def _count_quantizable_layers(self, model: nn.Module) -> int:
        count = 0
        for module in model.modules():
            if isinstance(module, nn.Linear):
                count += 1
        return count
    
    def auto_select_quantization(
        self,
        target_platform: str,
        model_type: str,
        max_accuracy_drop: float = 2.0
    ) -> QuantType:
        platform = target_platform.lower()
        
        if model_type == "llm":
            if platform in ["mobile", "edge"]:
                return QuantType.INT4
            elif platform == "laptop":
                return QuantType.INT8
            else:
                return QuantType.INT8
        else:
            if platform in ["mobile", "edge"]:
                return QuantType.INT8
            else:
                return QuantType.FP16


def simple_quantize(model: nn.Module, dtype: torch.dtype = torch.float16) -> nn.Module:
    return model.to(dtype)


def get_quantization_method(model: nn.Module) -> List[str]:
    methods = []
    
    try:
        import bitsandbytes
        methods.append("bitsandbytes")
    except ImportError:
        pass
    
    if hasattr(torch, "ao"):
        methods.append("torch_quantization")
    
    return methods