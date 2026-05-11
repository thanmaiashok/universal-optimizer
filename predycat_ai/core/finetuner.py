"""LoRA Fine-tuning for performance recovery"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class LoRAConfig:
    rank: int = 8
    alpha: int = 16
    target_modules: List[str] = None
    dropout: float = 0.05
    bias: str = "none"
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class LoRALayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int = 8, alpha: int = 16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
    def forward(self, x):
        return self.scaling * (x @ self.lora_A.T) @ self.lora_B.T


class LoRAWrapper(nn.Module):
    def __init__(self, base_model: nn.Module, config: LoRAConfig):
        super().__init__()
        self.base_model = base_model
        self.config = config
        self.lora_layers = nn.ModuleDict()
        
        self._apply_lora()
        
    def _apply_lora(self):
        for name, module in self.base_model.named_modules():
            if isinstance(module, nn.Linear):
                for target in self.config.target_modules:
                    if target in name:
                        lora = LoRALayer(
                            module.in_features,
                            module.out_features,
                            self.config.rank,
                            self.config.alpha
                        )
                        self.lora_layers[name] = lora
                        
    def forward(self, x):
        base_output = self.base_model(x)
        
        lora_output = 0
        for name, lora in self.lora_layers.items():
            if hasattr(lora, 'forward'):
                lora_output = lora(x)
                
        return base_output + lora_output


class LoRAFinetuner:
    def __init__(self, model: nn.Module, config: Optional[LoRAConfig] = None):
        self.model = model
        self.config = config or LoRAConfig()
        self.lora_model = None
        
    def apply_lora(self):
        logger.info(f"Applying LoRA with rank={self.config.rank}")
        self.lora_model = LoRAWrapper(self.model, self.config)
        return self.lora_model
    
    def finetune(
        self,
        prompts: List[str],
        tokenizer: Any,
        num_epochs: int = 3,
        lr: float = 1e-4
    ):
        if self.lora_model is None:
            self.lora_model = self.apply_lora()
            
        if tokenizer is None:
            logger.warning("No tokenizer, skipping fine-tuning")
            return self.lora_model
            
        optimizer = torch.optim.AdamW(self.lora_model.parameters(), lr=lr)
        
        for epoch in range(num_epochs):
            for prompt in prompts[:10]:
                try:
                    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
                    input_ids = inputs["input_ids"]
                    
                    optimizer.zero_grad()
                    outputs = self.lora_model(input_ids)
                    loss = outputs.mean()
                    
                    loss.backward()
                    optimizer.step()
                    
                except Exception as e:
                    logger.warning(f"Fine-tuning step failed: {e}")
                    
        return self.lora_model
    
    def merge_weights(self):
        if self.lora_model is None:
            return self.model
            
        for name, lora in self.lora_model.lora_layers.items():
            for base_name, base_module in self.lora_model.base_model.named_modules():
                if isinstance(base_module, nn.Linear) and base_name in name:
                    lora_B = base_module.weight.data
                    lora_A = lora.lora_A.data
                    rank = lora.rank
                    scale = lora.scaling
                    
                    update = (lora_A @ lora_B.T) * scale
                    base_module.weight.data += update[:base_module.weight.shape[0], :base_module.weight.shape[1]]
                    
        return self.lora_model.base_model


def quick_lora(model: nn.Module, rank: int = 8) -> LoRAWrapper:
    config = LoRAConfig(rank=rank)
    finetuner = LoRAFinetuner(model, config)
    return finetuner.apply_lora()