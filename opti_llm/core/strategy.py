"""Dynamic Strategy Selector - Auto-selects optimization pipeline"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class TargetPlatform(Enum):
    LAPTOP = "laptop"
    CLOUD = "cloud"
    MOBILE = "mobile"
    EDGE = "edge"


class ModelType(Enum):
    LLM = "llm"
    TRANSFORMER = "transformer"
    CNN = "cnn"
    RNN = "rnn"
    HYBRID = "hybrid"


class OptimizationPreset(Enum):
    ULTRA_COMPRESSION = "ultra_compression"
    BALANCED = "balanced"
    MAX_ACCURACY = "max_accuracy"
    MOBILE_SAFE = "mobile_safe"


class PipelineStep(Enum):
    QUANTIZE = "quantize"
    PRUNE = "prune"
    DISTILL = "distill"
    FINETUNE = "finetune"
    EXPORT = "export"


@dataclass
class PipelineConfig:
    steps: List[PipelineStep] = field(default_factory=list)
    quantize_type: str = "int8"
    prune_ratio: float = 0.2
    distill_layers: int = 0
    distill_hidden: int = 0
    batch_size: int = 1
    max_accuracy_drop: float = 2.0
    use_quantization_first: bool = True


@dataclass
class OptimizationStrategy:
    name: str
    platform: TargetPlatform
    preset: OptimizationPreset
    pipeline: PipelineConfig
    estimated_compression: float
    estimated_latency_reduction: float


_default_strategies = {
    (TargetPlatform.MOBILE, OptimizationPreset.BALANCED): PipelineConfig(
        steps=[PipelineStep.QUANTIZE, PipelineStep.DISTILL, PipelineStep.EXPORT],
        quantize_type="int4",
        prune_ratio=0.15,
        distill_layers=0.5,
        distill_hidden=0.5,
        batch_size=1,
        max_accuracy_drop=2.0
    ),
    (TargetPlatform.MOBILE, OptimizationPreset.ULTRA_COMPRESSION): PipelineConfig(
        steps=[PipelineStep.QUANTIZE, PipelineStep.PRUNE, PipelineStep.DISTILL, PipelineStep.EXPORT],
        quantize_type="int4",
        prune_ratio=0.3,
        distill_layers=0.6,
        distill_hidden=0.6,
        batch_size=1,
        max_accuracy_drop=3.0
    ),
    (TargetPlatform.MOBILE, OptimizationPreset.MAX_ACCURACY): PipelineConfig(
        steps=[PipelineStep.QUANTIZE, PipelineStep.DISTILL, PipelineStep.FINETUNE, PipelineStep.EXPORT],
        quantize_type="int8",
        prune_ratio=0.1,
        distill_layers=0.3,
        distill_hidden=0.3,
        batch_size=1,
        max_accuracy_drop=1.0
    ),
    (TargetPlatform.LAPTOP, OptimizationPreset.BALANCED): PipelineConfig(
        steps=[PipelineStep.QUANTIZE, PipelineStep.PRUNE, PipelineStep.EXPORT],
        quantize_type="fp16",
        prune_ratio=0.15,
        batch_size=8,
        max_accuracy_drop=2.0
    ),
    (TargetPlatform.LAPTOP, OptimizationPreset.ULTRA_COMPRESSION): PipelineConfig(
        steps=[PipelineStep.QUANTIZE, PipelineStep.PRUNE, PipelineStep.DISTILL, PipelineStep.EXPORT],
        quantize_type="int8",
        prune_ratio=0.25,
        distill_layers=0.4,
        distill_hidden=0.4,
        batch_size=4,
        max_accuracy_drop=2.5
    ),
    (TargetPlatform.CLOUD, OptimizationPreset.BALANCED): PipelineConfig(
        steps=[PipelineStep.QUANTIZE, PipelineStep.EXPORT],
        quantize_type="fp16",
        batch_size=16,
        max_accuracy_drop=1.5
    ),
    (TargetPlatform.CLOUD, OptimizationPreset.MAX_ACCURACY): PipelineConfig(
        steps=[PipelineStep.QUANTIZE, PipelineStep.EXPORT],
        quantize_type="fp16",
        batch_size=32,
        max_accuracy_drop=0.5
    ),
    (TargetPlatform.EDGE, OptimizationPreset.BALANCED): PipelineConfig(
        steps=[PipelineStep.QUANTIZE, PipelineStep.PRUNE, PipelineStep.EXPORT],
        quantize_type="int8",
        prune_ratio=0.2,
        batch_size=1,
        max_accuracy_drop=2.0
    ),
}


class StrategySelector:
    def __init__(self):
        self.strategies = _default_strategies.copy()
        
    def select_strategy(
        self,
        platform: TargetPlatform,
        model_type: ModelType,
        preset: OptimizationPreset = OptimizationPreset.BALANCED,
        constraints: Optional[Dict[str, Any]] = None
    ) -> PipelineConfig:
        logger.info(f"Selecting strategy for {platform.value} with {preset.value}")
        
        key = (platform, preset)
        
        if key in self.strategies:
            config = self.strategies[key]
        else:
            config = self._generate_strategy(platform, model_type, preset, constraints)
        
        config = self._adjust_for_constraints(config, constraints)
        
        return config
    
    def _generate_strategy(
        self,
        platform: TargetPlatform,
        model_type: ModelType,
        preset: OptimizationPreset,
        constraints: Optional[Dict[str, Any]]
    ) -> PipelineConfig:
        if platform == TargetPlatform.MOBILE:
            if model_type == ModelType.LLM:
                return PipelineConfig(
                    steps=[PipelineStep.QUANTIZE, PipelineStep.DISTILL, PipelineStep.EXPORT],
                    quantize_type="int4",
                    max_accuracy_drop=2.0
                )
            else:
                return PipelineConfig(
                    steps=[PipelineStep.QUANTIZE, PipelineStep.EXPORT],
                    quantize_type="int8",
                    max_accuracy_drop=2.0
                )
        elif platform == TargetPlatform.LAPTOP:
            return PipelineConfig(
                steps=[PipelineStep.QUANTIZE, PipelineStep.PRUNE, PipelineStep.EXPORT],
                quantize_type="fp16",
                prune_ratio=0.15,
                max_accuracy_drop=2.0
            )
        elif platform == TargetPlatform.CLOUD:
            return PipelineConfig(
                steps=[PipelineStep.QUANTIZE, PipelineStep.EXPORT],
                quantize_type="fp16",
                max_accuracy_drop=1.0
            )
        else:
            return PipelineConfig(
                steps=[PipelineStep.QUANTIZE, PipelineStep.EXPORT],
                quantize_type="int8",
                max_accuracy_drop=2.0
            )
    
    def _adjust_for_constraints(
        self,
        config: PipelineConfig,
        constraints: Optional[Dict[str, Any]]
    ) -> PipelineConfig:
        if not constraints:
            return config
        
        if "max_size_mb" in constraints:
            target_size = constraints["max_size_mb"]
            if target_size is not None and target_size < 500:
                config.quantize_type = "int4"
                config.prune_ratio = min(0.4, config.prune_ratio + 0.1)

        if "max_ram_mb" in constraints:
            max_ram = constraints["max_ram_mb"]
            if max_ram is not None and max_ram < 2048:
                config.batch_size = 1
                config.quantize_type = "int4"
        
        if "max_accuracy_drop" in constraints:
            config.max_accuracy_drop = constraints["max_accuracy_drop"]
        
        return config
    
    def get_pipeline_steps(self, config: PipelineConfig) -> List[str]:
        return [step.value for step in config.steps]
    
    def get_steps_description(self, config: PipelineConfig) -> str:
        descriptions = []
        
        for step in config.steps:
            if step == PipelineStep.QUANTIZE:
                desc = f"Quantize to {config.quantize_type}"
            elif step == PipelineStep.PRUNE:
                desc = f"Prune {int(config.prune_ratio * 100)}% of weights"
            elif step == PipelineStep.DISTILL:
                desc = f"Distill to {config.distill_hidden * 100}% size"
            elif step == PipelineStep.FINETUNE:
                desc = "LoRA fine-tuning"
            elif step == PipelineStep.EXPORT:
                desc = "Export model"
            else:
                desc = step.value
                
            descriptions.append(desc)
        
        return " → ".join(descriptions)


def auto_detect_platform() -> TargetPlatform:
    try:
        import torch
        if torch.cuda.is_available():
            return TargetPlatform.LAPTOP
    except:
        pass
    
    try:
        import psutil
        mem = psutil.virtual_memory()
        if mem.total < 8 * (1024**3):
            return TargetPlatform.MOBILE
    except:
        pass
    
    return TargetPlatform.LAPTOP


def quick_select_pipeline(target: str) -> PipelineConfig:
    platform = TargetPlatform(target.lower()) if target.lower() in [p.value for p in TargetPlatform] else TargetPlatform.LAPTOP
    selector = StrategySelector()
    return selector.select_strategy(platform, ModelType.LLM)