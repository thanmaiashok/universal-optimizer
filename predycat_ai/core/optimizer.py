"""PredycatAI Main Orchestrator - Universal Optimizer Engine"""

import os
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from datetime import datetime

import torch
import torch.nn as nn

from .loader import UniversalLoader, ModelInfo
from .profiler import MultiProfiler, ProfileResult, Platform
from .quantizer import QuantizationEngine, QuantType
from .pruner import AdaptivePruner
from .distiller import DistillationEngine
from .evaluator import ReasoningEvaluator, AccuracyMonitor
from .strategy import StrategySelector, PipelineConfig, TargetPlatform, OptimizationPreset
from .exporter import ExportEngine, ExportFormat

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    original_size_mb: float
    optimized_size_mb: float
    compression_ratio: float
    original_latency_ms: float
    optimized_latency_ms: float
    latency_improvement: float
    original_vram_mb: float
    optimized_vram_mb: float
    original_accuracy: float
    optimized_accuracy: float
    accuracy_drop: float
    pipeline_steps: List[str]
    export_formats: List[str]
    output_paths: List[str]
    success: bool
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class OptimizationConfig:
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


class PredycatOptimizer:
    def __init__(
        self,
        model: Any = None,
        tokenizer: Any = None,
        config: Optional[OptimizationConfig] = None,
        output_dir: str = "./outputs"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or OptimizationConfig()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.loader = UniversalLoader()
        self.profiler = MultiProfiler()
        self.quantizer = QuantizationEngine()
        self.pruner = AdaptivePruner()
        self.distiller = None
        self.evaluator = ReasoningEvaluator(model, tokenizer)
        self.strategy_selector = StrategySelector()
        self.exporter = ExportEngine(str(self.output_dir))
        self.accuracy_monitor = AccuracyMonitor(threshold=config.max_accuracy_drop if config else 2.0)
        
        self.original_model = None
        self.optimized_model = None
        self.model_info = None
        self.results = []
        
        logger.info("PredycatAI Optimizer initialized")
        
    def optimize(
        self,
        model: Any,
        model_path: Optional[str] = None,
        **kwargs
    ) -> OptimizationResult:
        start_time = time.time()
        
        self.original_model = model
        
        logger.info(f"Starting optimization for target: {self.config.target_platform}")
        
        if isinstance(self.config.target_platform, str):
            platform = TargetPlatform(self.config.target_platform.lower())
        else:
            platform = self.config.target_platform
        
        try:
            self.model_info = self._load_model(model, model_path)
            
            profile_before = self._profile_original()
            
            strategy = self.strategy_selector.select_strategy(
                platform=platform,
                model_type=self._detect_model_type(),
                preset=OptimizationPreset(self.config.preset.lower()),
                constraints={
                    "max_size_mb": self.config.max_size_mb,
                    "max_ram_mb": self.config.max_ram_mb,
                    "max_accuracy_drop": self.config.max_accuracy_drop
                }
            )
            
            logger.info(f"Pipeline: {strategy.steps}")
            
            self.optimized_model = model
            
            if self.config.use_quantization:
                self.optimized_model = self._apply_quantization(strategy)
            
            if self.config.use_pruning and platform in [TargetPlatform.MOBILE, TargetPlatform.EDGE]:
                self.optimized_model = self._apply_pruning(strategy)
            
            if self.config.use_distillation and platform == TargetPlatform.MOBILE:
                self.optimized_model = self._apply_distillation(strategy)
            
            profile_after = self._profile_optimized()
            
            if self.config.enable_reasoning_check:
                self._verify_accuracy()
            
            output_paths = self._export_model()
            
            elapsed = time.time() - start_time
            
            result = OptimizationResult(
                original_size_mb=profile_before.get("size_mb", 0),
                optimized_size_mb=profile_after.get("size_mb", 0),
                compression_ratio=profile_before.get("size_mb", 1) / max(profile_after.get("size_mb", 0.1), 0.1),
                original_latency_ms=profile_before.get("latency_ms", 0),
                optimized_latency_ms=profile_after.get("latency_ms", 0),
                latency_improvement=max(0.0, (profile_before.get("latency_ms", 0) - profile_after.get("latency_ms", 0)) / max(profile_before.get("latency_ms", 1), 0.001) * 100),
                original_vram_mb=profile_before.get("vram_mb", 0),
                optimized_vram_mb=profile_after.get("vram_mb", 0),
                original_accuracy=profile_before.get("accuracy", 0),
                optimized_accuracy=profile_after.get("accuracy", 0),
                accuracy_drop=abs(profile_before.get("accuracy", 0) - profile_after.get("accuracy", 0)),
                pipeline_steps=[s.value for s in strategy.steps],
                export_formats=self.config.export_formats,
                output_paths=output_paths,
                success=True,
                metadata={
                    "platform": platform.value,
                    "preset": self.config.preset,
                    "elapsed_time_sec": elapsed,
                    "strategy": asdict(strategy)
                }
            )
            
            self._save_report(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            
            return OptimizationResult(
                original_size_mb=0,
                optimized_size_mb=0,
                compression_ratio=0,
                original_latency_ms=0,
                optimized_latency_ms=0,
                latency_improvement=0,
                original_vram_mb=0,
                optimized_vram_mb=0,
                original_accuracy=0,
                optimized_accuracy=0,
                accuracy_drop=0,
                pipeline_steps=[],
                export_formats=[],
                output_paths=[],
                success=False,
                error=str(e)
            )
    
    def _load_model(self, model: Any, model_path: Optional[str] = None) -> ModelInfo:
        if model is not None:
            if hasattr(model, "config"):
                return ModelInfo(
                    source=self.loader.detect_source(model_path or "unknown"),
                    architecture=self.loader.detect_architecture(model.config.to_dict() if hasattr(model.config, "to_dict") else {}),
                    num_parameters=sum(p.numel() for p in model.parameters()),
                    num_layers=getattr(model.config, "num_hidden_layers", 0)
                )
        return ModelInfo(
            source=self.loader.detect_source(model_path or "unknown"),
            architecture=self.loader.detect_architecture({}),
            num_parameters=0,
            num_layers=0
        )
    
    def _detect_model_type(self):
        if self.model_info:
            return self.model_info.architecture
        return "unknown"
    
    def _measure_latency_ms(self, model) -> float:
        import time
        import torch
        try:
            model.eval()
            device = next(model.parameters()).device
            dummy = torch.randint(0, 100, (1, 16)).to(device)
            with torch.no_grad():
                for _ in range(3):
                    model(input_ids=dummy)
            times = []
            with torch.no_grad():
                for _ in range(10):
                    t0 = time.perf_counter()
                    model(input_ids=dummy)
                    times.append((time.perf_counter() - t0) * 1000)
            return round(sum(times) / len(times), 3)
        except Exception as e:
            logger.warning(f"Latency measure failed: {e}")
            return 0.0

    def _measure_perplexity(self, model) -> float:
        import torch
        import math
        # fixed realistic token IDs (common English words in typical LLM vocabs)
        _FIXED_TOKENS = [464, 995, 318, 257, 1263, 290, 6496, 1295, 13, 383,
                         2068, 3290, 9229, 625, 262, 16931, 9371, 11, 475, 262,
                         3057, 318, 257, 4171, 530, 13, 383, 1281, 6952, 281,
                         4939, 523, 326, 477, 460, 910, 7494, 3276, 1494, 13,
                         198, 1026, 373, 257, 6016, 290, 4362, 1110, 13, 383,
                         4252, 2630, 287, 262, 6921, 287, 257, 4991, 286, 1657]
        try:
            model.eval()
            device = next(model.parameters()).device
            tokens = torch.tensor([_FIXED_TOKENS], dtype=torch.long).to(device)
            with torch.no_grad():
                out = model(input_ids=tokens, labels=tokens)
            if hasattr(out, "loss") and out.loss is not None:
                return min(math.exp(out.loss.item()), 9999.0)
            logits = getattr(out, "logits", None) or getattr(out, "last_hidden_state", None)
            if logits is not None:
                probs = torch.softmax(logits.float(), dim=-1)
                entropy = -(probs * (probs + 1e-9).log()).sum(-1).mean().item()
                return max(1.0, entropy)
            return 0.0
        except Exception as e:
            logger.warning(f"Perplexity measure failed: {e}")
            return 0.0

    _PPL_INCREASE_WARN_THRESHOLD = 0.15   # 15% PPL increase → hallucination risk warning
    _PPL_INCREASE_HARD_THRESHOLD = 0.40   # 40% PPL increase → compression too aggressive

    def _profile_original(self) -> Dict[str, float]:
        profile = {"size_mb": 0, "latency_ms": 0, "vram_mb": 0, "accuracy": 0, "perplexity": 0}
        if self.original_model:
            try:
                params = sum(p.numel() for p in self.original_model.parameters())
                profile["size_mb"] = params * 4 / (1024 ** 2)
                profile["latency_ms"] = self._measure_latency_ms(self.original_model)
                ppl = self._measure_perplexity(self.original_model)
                self._original_ppl = ppl  # store for post-compression comparison
                profile["perplexity"] = round(ppl, 3)
                profile["accuracy"] = round(100 - min(ppl / 10, 50), 2) if ppl > 0 else 0
                logger.info(f"Original model: {profile}")
            except Exception as e:
                logger.warning(f"Profile failed: {e}")
        return profile

    def _profile_optimized(self) -> Dict[str, float]:
        profile = {"size_mb": 0, "latency_ms": 0, "vram_mb": 0, "accuracy": 0, "perplexity": 0}
        if self.optimized_model:
            try:
                params = sum(p.numel() for p in self.optimized_model.parameters())
                try:
                    element_size = next(self.optimized_model.parameters()).element_size()
                except Exception:
                    element_size = 4
                profile["size_mb"] = params * element_size / (1024 ** 2)
                profile["latency_ms"] = self._measure_latency_ms(self.optimized_model)
                ppl = self._measure_perplexity(self.optimized_model)
                profile["perplexity"] = round(ppl, 3)
                profile["accuracy"] = round(100 - min(ppl / 10, 50), 2) if ppl > 0 else 0
                logger.info(f"Optimized model: {profile}")

                # PPL gate: compare to original to catch hallucination-inducing compression
                orig_ppl = getattr(self, "_original_ppl", 0)
                if orig_ppl > 0 and ppl > 0:
                    ppl_increase = (ppl - orig_ppl) / orig_ppl
                    profile["ppl_increase_pct"] = round(ppl_increase * 100, 1)
                    if ppl_increase > self._PPL_INCREASE_HARD_THRESHOLD:
                        logger.error(
                            f"HALLUCINATION RISK HIGH: PPL increased {ppl_increase*100:.1f}% "
                            f"({orig_ppl:.1f} → {ppl:.1f}). Compression too aggressive — "
                            f"consider lower pruning ratio or higher precision quantization."
                        )
                    elif ppl_increase > self._PPL_INCREASE_WARN_THRESHOLD:
                        logger.warning(
                            f"HALLUCINATION RISK ELEVATED: PPL increased {ppl_increase*100:.1f}% "
                            f"({orig_ppl:.1f} → {ppl:.1f}). Model may hallucinate more than baseline."
                        )
                    else:
                        logger.info(f"PPL gate OK: increase {ppl_increase*100:.1f}% within safe range.")

            except Exception as e:
                logger.warning(f"Profile failed: {e}")
        return profile
    
    def _apply_quantization(self, strategy: PipelineConfig) -> nn.Module:
        logger.info(f"Applying quantization: {strategy.quantize_type}")
        
        if not self.optimized_model:
            return self.optimized_model
            
        try:
            quant_type = QuantType(strategy.quantize_type)
            self.optimized_model, quant_result = self.quantizer.quantize(
                self.optimized_model, 
                quant_type=quant_type,
                max_accuracy_drop=strategy.max_accuracy_drop
            )
            logger.info(f"Quantization result: {quant_result.compression_ratio:.2f}x compression")
        except Exception as e:
            logger.warning(f"Quantization failed: {e}")
            
        return self.optimized_model
    
    def _apply_pruning(self, strategy: PipelineConfig) -> nn.Module:
        logger.info(f"Applying pruning: {strategy.prune_ratio}")
        
        if not self.optimized_model:
            return self.optimized_model
            
        try:
            self.optimized_model, prune_result = self.pruner.prune_model(
                self.optimized_model,
                pruning_ratio=strategy.prune_ratio,
                max_accuracy_drop=strategy.max_accuracy_drop
            )
            logger.info(f"Pruning result: {prune_result.compression_ratio:.2f}x")
        except Exception as e:
            logger.warning(f"Pruning failed: {e}")
            
        return self.optimized_model
    
    def _apply_distillation(self, strategy: PipelineConfig) -> nn.Module:
        logger.info("Applying distillation")
        
        if not self.original_model or not self.optimized_model:
            return self.optimized_model
            
        if self.distiller is None:
            self.distiller = DistillationEngine(self.original_model)
            
        try:
            self.optimized_model, dist_result = self.distiller.distill(
                self.optimized_model,
                self.original_model,
                num_epochs=2
            )
            logger.info(f"Distillation complete")
        except Exception as e:
            logger.warning(f"Distillation failed: {e}")
            
        return self.optimized_model
    
    def _verify_accuracy(self):
        logger.info("Verifying accuracy")
        
        if not self.optimized_model or not self.evaluator:
            return
            
        try:
            score = self.evaluator.evaluate_reasoning(self.optimized_model)
            logger.info(f"Reasoning score: {score.overall_score:.2f}")
        except Exception as e:
            logger.warning(f"Accuracy verification failed: {e}")
    
    def _export_model(self) -> List[str]:
        paths = []
        
        for fmt in self.config.export_formats:
            try:
                fmt_enum = ExportFormat(fmt.lower())
                result = self.exporter.export(
                    self.optimized_model,
                    format=fmt_enum,
                    model_name=f"predycat_optimized"
                )
                if result.success:
                    paths.append(result.output_path)
                    logger.info(f"Exported to {result.output_path}")
            except Exception as e:
                logger.warning(f"Export failed for {fmt}: {e}")
                
        return paths
    
    def _save_report(self, result: OptimizationResult):
        report = {
            "timestamp": datetime.now().isoformat(),
            "config": asdict(self.config),
            "result": asdict(result)
        }
        
        report_path = self.output_dir / "optimization_report.json"

        def _default(obj):
            if hasattr(obj, "value"):
                return obj.value  # Enum → its value
            return str(obj)

        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=_default)
            
        logger.info(f"Report saved to {report_path}")
        return str(report_path)


def optimize_model(
    model_path: str,
    target: str = "mobile",
    preset: str = "balanced",
    max_drop: float = 2.0,
    output_dir: str = "./outputs"
) -> OptimizationResult:
    config = OptimizationConfig(
        target_platform=target,
        preset=preset,
        max_accuracy_drop=max_drop
    )
    
    optimizer = PredycatOptimizer(config=config, output_dir=output_dir)
    
    result = optimizer.optimize(model_path)
    
    return result


def quick_optimize(
    model: Any,
    target: str = "mobile"
) -> OptimizationResult:
    config = OptimizationConfig(target_platform=target)
    
    optimizer = PredycatOptimizer(config=config)
    
    return optimizer.optimize(model)


