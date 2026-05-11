"""Self-Learning Optimizer - Learns from past optimizations"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class OptimizationRecord:
    model_type: str
    target_platform: str
    preset: str
    pipeline: List[str]
    compression_ratio: float
    latency_improvement: float
    accuracy_drop: float
    success: bool
    timestamp: str


class SelfLearningOptimizer:
    """Learns from past runs to improve future optimizations"""
    
    def __init__(self, memory_path: str = "./optimization_memory.json"):
        self.memory_path = Path(memory_path)
        self.records: List[OptimizationRecord] = []
        self.success_patterns = defaultdict(list)
        self.failure_patterns = defaultdict(list)
        self._load_memory()
        
    def _load_memory(self):
        if self.memory_path.exists():
            try:
                with open(self.memory_path) as f:
                    data = json.load(f)
                    for item in data.get("records", []):
                        rec = OptimizationRecord(**item)
                        self.records.append(rec)
                        if rec.success:
                            key = (rec.model_type, rec.target_platform)
                            self.success_patterns[key].append(rec)
                        else:
                            key = (rec.model_type, rec.target_platform)
                            self.failure_patterns[key].append(rec)
                logger.info(f"Loaded {len(self.records)} optimization records")
            except Exception as e:
                logger.warning(f"Failed to load memory: {e}")
                
    def _save_memory(self):
        try:
            data = {
                "records": [asdict(r) for r in self.records],
                "updated": datetime.now().isoformat()
            }
            with open(self.memory_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save memory: {e}")
    
    def record_result(
        self,
        model_type: str,
        target_platform: str,
        preset: str,
        pipeline: List[str],
        compression: float,
        latency: float,
        accuracy_drop: float,
        success: bool
    ):
        record = OptimizationRecord(
            model_type=model_type,
            target_platform=target_platform,
            preset=preset,
            pipeline=pipeline,
            compression_ratio=compression,
            latency_improvement=latency,
            accuracy_drop=accuracy_drop,
            success=success,
            timestamp=datetime.now().isoformat()
        )
        
        self.records.append(record)
        
        key = (model_type, target_platform)
        if success:
            self.success_patterns[key].append(record)
        else:
            self.failure_patterns[key].append(record)
            
        self._save_memory()
        logger.info(f"Recorded optimization: {model_type} -> {target_platform}")
    
    def get_best_pipeline(
        self,
        model_type: str,
        target_platform: str
    ) -> Dict[str, Any]:
        key = (model_type, target_platform)
        
        if key in self.success_patterns:
            records = self.success_patterns[key]
            
            best = max(records, key=lambda r: r.compression_ratio * r.latency_improvement)
            
            return {
                "preset": best.preset,
                "pipeline": best.pipeline,
                "confidence": len(records) / max(1, len(self.records)),
                "avg_compression": sum(r.compression_ratio for r in records) / len(records),
                "avg_latency": sum(r.latency_improvement for r in records) / len(records)
            }
        
        return {"preset": "balanced", "pipeline": ["quantize", "export"], "confidence": 0.0}
    
    def get_warnings(
        self,
        model_type: str,
        target_platform: str
    ) -> List[str]:
        warnings = []
        key = (model_type, target_platform)
        
        if key in self.failure_patterns:
            failures = self.failure_patterns[key]
            for f in failures[-3:]:
                warnings.append(f"Previous failure: {f.pipeline}")
                
        return warnings
    
    def suggest_parameters(
        self,
        model_type: str,
        target_platform: str,
        model_size_mb: float
    ) -> Dict[str, Any]:
        best = self.get_best_pipeline(model_type, target_platform)
        warnings = self.get_warnings(model_type, target_platform)
        
        params = {
            "suggested_preset": best.get("preset", "balanced"),
            "suggested_pipeline": best.get("pipeline", ["quantize"]),
            "confidence": best.get("confidence", 0.0),
            "past_experiences": len(self.records),
            "warnings": warnings
        }
        
        if model_size_mb > 1000:
            params["suggested_pipeline"].insert(0, "quantize")
            params["suggested_pipeline"].insert(1, "prune")
            
        return params


def get_default_parameters(
    model_type: str,
    target_platform: str
) -> Dict[str, Any]:
    """Get conservative defaults with no learning"""
    
    presets = {
        ("llm", "mobile"): {
            "preset": "mobile_safe",
            "quantize_type": "int4",
            "prune_ratio": 0.15,
            "distill": True
        },
        ("llm", "laptop"): {
            "preset": "balanced",
            "quantize_type": "int8",
            "prune_ratio": 0.1,
            "distill": False
        },
        ("cnn", "mobile"): {
            "preset": "mobile_safe", 
            "quantize_type": "int8",
            "prune_ratio": 0.2,
            "distill": False
        }
    }
    
    return presets.get((model_type.lower(), target_platform.lower()), {
        "preset": "balanced",
        "quantize_type": "fp16",
        "prune_ratio": 0.1,
        "distill": False
    })


def clear_memory(path: str = "./optimization_memory.json"):
    """Clear learning memory"""
    p = Path(path)
    if p.exists():
        p.unlink()
        logger.info("Memory cleared")