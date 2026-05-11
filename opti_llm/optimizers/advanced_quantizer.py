"""Enhanced Quantization with bitsandbytes support"""

import logging
from typing import Optional, Tuple
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class BitsAndBytesQuantizer:
    """4-bit quantization using bitsandbytes"""
    
    def __init__(self, load_in_4bit: bool = True, bnb_4bit_compute_dtype: str = "float16"):
        self.load_in_4bit = load_in_4bit
        self.compute_dtype = bnb_4bit_compute_dtype
        self.available = self._check_available()
        
    def _check_available(self) -> bool:
        try:
            import bitsandbytes
            return True
        except ImportError:
            return False
            
    def quantize_model(self, model: nn.Module) -> nn.Module:
        if not self.available:
            logger.warning("bitsandbytes not installed")
            return model
            
        try:
            import bitsandbytes as bnb
            
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    if module.out_features >= 128:
                        new_module = bnb.nn.Linear4bit(
                            module.out_features,
                            module.in_features,
                            bias=module.bias is not None,
                            has_fp16_weights=False,
                            memory_efficient=True
                        )
                        new_module.weight.data = module.weight.data.clone()
                        if module.bias is not None:
                            new_module.bias.data = module.bias.data.clone()
                        setattr(model, name, new_module)
                        
            logger.info("Applied 4-bit quantization")
            return model
            
        except Exception as e:
            logger.warning(f"BNB quantization failed: {e}")
            return model
            
    def get_memory_savings(self, model: nn.Module) -> dict:
        original = sum(p.numel() * p.element_size() for p in model.parameters())
        estimated = original / 4
        return {
            "original_mb": original / (1024**2),
            "estimated_mb": estimated / (1024**2),
            "savings_percent": (1 - estimated/original) * 100
        }


class AWQQuantizer:
    """Activation-Aware Quantization for LLMs"""
    
    def __init__(self):
        self.available = self._check_available()
        
    def _check_available(self) -> bool:
        try:
            from awq import AutoAWQ
            return True
        except ImportError:
            return False
            
    def quantize(self, model, tokenizer, quant_config: dict = None) -> Tuple[nn.Module, str]:
        if not self.available:
            raise ImportError("AWQ not installed: pip install awq")
            
        from awq import AutoAWQ
        
        awq_model = AutoAWQ.inject(model, quant_config or {"w_bit": 4, "group_size": 128})
        return awq_model, "awq quantized"


class GPTQQuantizer:
    """GPTQ post-training quantization"""
    
    def __init__(self):
        self.available = self._check_available()
        
    def _check_available(self) -> bool:
        try:
            from auto_gptq import AutoGPTQForCausalLM
            return True
        except ImportError:
            return False
            
    def quantize(self, model, tokenizer, bits: int = 4, group_size: int = 128):
        if not self.available:
            raise ImportError("GPTQ not installed: pip install auto-gptq")
            
        from auto_gptq import AutoGPTQForCausalLM
        
        quantized_model = AutoGPTQForCausalLM.quantize_model(
            model, bits=bits, group_size=group_size
        )
        return quantized_model


def get_quantizer(backend: str = "auto") -> Tuple[object, str]:
    """Auto-select best quantizer"""
    backends = {
        "bnb": (BitsAndBytesQuantizer, "bitsandbytes"),
        "awq": (AWQQuantizer, "AWQ"),
        "gptq": (GPTQQuantizer, "GPTQ"),
        "default": (BitsAndBytesQuantizer, "bitsandbytes"),
    }
    
    if backend == "auto":
        for name, (cls, pkg) in backends.items():
            if pkg == "bitsandbytes":
                inst = cls()
                if inst.available:
                    return inst, name
        return BitsAndBytesQuantizer(), "fallback"
    
    cls, pkg = backends.get(backend, (BitsAndBytesQuantizer, "bitsandbytes"))
    return cls(), backend