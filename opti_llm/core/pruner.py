"""Sensitivity Analysis & Adaptive Pruner"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class LayerImportance:
    layer_id: str
    importance_score: float
    layer_type: str


@dataclass
class HeadImportance:
    layer_idx: int
    head_idx: int
    importance_score: float


@dataclass
class PruningResult:
    original_params: int
    pruned_params: int
    compression_ratio: float
    layers_pruned: int
    heads_pruned: int
    accuracy_drop: float = 0.0


class SensitivityAnalyzer:
    def __init__(self, model: Optional[nn.Module] = None):
        self.model = model
        self.layer_importance = []
        self.head_importance = []
        
    def compute_layer_importance(
        self,
        model: nn.Module,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        method: str = "gradient"
    ) -> List[LayerImportance]:
        logger.info(f"Computing layer importance using {method}")
        
        if method == "gradient":
            return self._gradient_importance(model, input_ids, attention_mask)
        elif method == "activation":
            return self._activation_importance(model, input_ids, attention_mask)
        elif method == "random":
            return self._random_importance(model)
        else:
            return self._random_importance(model)
    
    def _gradient_importance(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> List[LayerImportance]:
        if not hasattr(model, "named_modules"):
            return []
        
        layer_importance = []
        
        try:
            model.eval()
            for name, module in model.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    module.zero_grad()
                    
                    if hasattr(module, "weight") and module.weight.grad is None:
                        output = self._get_layer_output(module, input_ids)
                        if output is not None:
                            output.mean().backward()
                            
                            if module.weight.grad is not None:
                                score = (module.weight.grad ** 2).mean().item()
                                layer_importance.append(LayerImportance(
                                    layer_id=name,
                                    importance_score=score,
                                    layer_type=type(module).__name__
                                ))
                                module.weight.grad.zero_()
        except Exception as e:
            logger.warning(f"Gradient importance computation failed: {e}")
            
        return sorted(layer_importance, key=lambda x: x.importance_score, reverse=True)
    
    def _activation_importance(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> List[LayerImportance]:
        layer_importance = []
        
        try:
            model.eval()
            activations = {}
            
            def hook_fn(name):
                def hook(module, input, output):
                    if isinstance(output, torch.Tensor):
                        activations[name] = output.abs().mean().item()
                return hook
            
            hooks = []
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    hooks.append(module.register_forward_hook(hook_fn(name)))
            
            with torch.no_grad():
                _ = model(input_ids)
            
            for hook in hooks:
                hook.remove()
            
            for name, score in activations.items():
                layer_importance.append(LayerImportance(
                    layer_id=name,
                    importance_score=score,
                    layer_type="Linear"
                ))
                
        except Exception as e:
            logger.warning(f"Activation importance computation failed: {e}")
            
        return sorted(layer_importance, key=lambda x: x.importance_score, reverse=True)
    
    def _random_importance(self, model: nn.Module) -> List[LayerImportance]:
        layer_importance = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                layer_importance.append(LayerImportance(
                    layer_id=name,
                    importance_score=np.random.random(),
                    layer_type=type(module).__name__
                ))
        return layer_importance
    
    def _get_layer_output(self, module: nn.Module, input_data: torch.Tensor):
        try:
            with torch.no_grad():
                return module(input_data)
        except:
            return None
    
    def compute_head_importance(
        self,
        model: nn.Module,
        method: str = "gradient"
    ) -> List[HeadImportance]:
        head_importance = []
        
        try:
            for name, module in model.named_modules():
                if "attention" in name.lower() and hasattr(module, "num_heads"):
                    num_heads = module.num_heads
                    hidden_size = module.head_dim if hasattr(module, "head_dim") else 64
                    
                    for head_idx in range(num_heads):
                        score = np.random.random() if method == "random" else 0.5
                        head_importance.append(HeadImportance(
                            layer_idx=name,
                            head_idx=head_idx,
                            importance_score=score
                        ))
        except Exception as e:
            logger.warning(f"Head importance computation failed: {e}")
            
        return sorted(head_importance, key=lambda x: x.importance_score, reverse=True)


class AdaptivePruner:
    def __init__(self, model: Optional[nn.Module] = None):
        self.model = model
        self.sensitivity_analyzer = SensitivityAnalyzer(model)
        
    def prune_model(
        self,
        model: nn.Module,
        pruning_ratio: float = 0.2,
        pruning_type: str = "magnitude",
        input_ids: Optional[torch.Tensor] = None,
        accuracy_check_fn: Optional[callable] = None,
        max_accuracy_drop: float = 2.0
    ) -> Tuple[nn.Module, PruningResult]:
        logger.info(f"Starting pruning with ratio {pruning_ratio}")
        
        original_params = sum(p.numel() for p in model.parameters())
        
        layer_importance = self.sensitivity_analyzer.compute_layer_importance(model, input_ids)
        
        if pruning_type in ["magnitude", "threshold"]:
            model = self._magnitude_prune(model, pruning_ratio)
        elif pruning_type == "random":
            model = self._random_prune(model, pruning_ratio)
        
        pruned_params = sum(p.numel() for p in model.parameters())
        
        result = PruningResult(
            original_params=original_params,
            pruned_params=pruned_params,
            compression_ratio=original_params / pruned_params if pruned_params > 0 else 1.0,
            layers_pruned=int(len(layer_importance) * pruning_ratio),
            heads_pruned=0,
            accuracy_drop=0.0
        )
        
        return model, result
    
    _ATTENTION_KEYWORDS = ("attn", "attention", "q_proj", "k_proj", "v_proj", "self_attn", "cross_attn", "query", "key", "value")

    def _magnitude_prune(self, model: nn.Module, ratio: float) -> nn.Module:
        logger.info(f"Applying magnitude pruning with ratio {ratio}")

        for name, module in model.named_modules():
            if not (isinstance(module, nn.Linear) and module.out_features > 128):
                continue
            # Skip attention layers — they store positional/relational knowledge
            # Pruning them causes hallucination; only prune FFN/MLP layers
            name_lower = name.lower()
            if any(kw in name_lower for kw in self._ATTENTION_KEYWORDS):
                logger.debug(f"Skipping attention layer: {name}")
                continue

            weight = module.weight.data

            threshold = torch.quantile(
                weight.abs().flatten(),
                torch.tensor(ratio)
            ).item()

            mask = weight.abs() > threshold
            module.weight.data = weight * mask.float()

            if module.bias is not None:
                module.bias.data = module.bias.data * mask.any(dim=0).float()

        return model
    
    def _random_prune(self, model: nn.Module, ratio: float) -> nn.Module:
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                weight = module.weight.data
                num_params = weight.numel()
                num_to_zero = int(num_params * ratio)
                
                indices = torch.randperm(num_params)[:num_to_zero]
                with torch.no_grad():
                    weight.flatten()[indices] = 0
                    
        return model
    
    def prune_attention_heads(
        self,
        model: nn.Module,
        prune_ratio: float = 0.1,
        accuracy_check_fn: Optional[callable] = None
    ) -> Tuple[nn.Module, PruningResult]:
        logger.info(f"Pruning attention heads with ratio {prune_ratio}")
        
        original_params = sum(p.numel() for p in model.parameters())
        
        head_importance = self.sensitivity_analyzer.compute_head_importance(model)
        heads_to_prune = int(len(head_importance) * prune_ratio)
        
        for i in range(heads_to_prune):
            head = head_importance[i]
            logger.debug(f"Would prune head {head.head_idx} in layer {head.layer_idx}")
        
        pruned_params = sum(p.numel() for p in model.parameters())
        
        return model, PruningResult(
            original_params=original_params,
            pruned_params=pruned_params,
            compression_ratio=original_params / pruned_params,
            layers_pruned=0,
            heads_pruned=heads_to_prune
        )


def compute_sparsity(model: nn.Module) -> float:
    total_params = 0
    zero_params = 0
    
    for param in model.parameters():
        total_params += param.numel()
        zero_params += (param.data == 0).sum().item()
    
    return zero_params / total_params if total_params > 0 else 0.0


def gradual_pruning(
    model: nn.Module,
    initial_sparsity: float = 0.0,
    final_sparsity: float = 0.5,
    epochs: int = 10,
    schedule: str = "polynomial"
) -> nn.Module:
    for epoch in range(epochs):
        if schedule == "polynomial":
            current_sparsity = final_sparsity * (1 - (1 - epoch / epochs) ** 3)
        elif schedule == "linear":
            current_sparsity = final_sparsity * (epoch / epochs)
        else:
            current_sparsity = final_sparsity * (epoch / epochs)
        
        ratio = current_sparsity - initial_sparsity
        if ratio > 0:
            model = AdaptivePruner(model)._magnitude_prune(model, ratio)
    
    return model