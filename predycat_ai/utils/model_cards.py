"""Model Cards - Track model metadata and lineage"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ModelCard:
    model_id: str
    name: str
    base_model: str
    optimization_target: str
    created_at: str
    metrics: Dict[str, float]
    pipeline_steps: list
    config: Dict[str, Any]
    lineage: list = None
    
    def __post_init__(self):
        if self.lineage is None:
            self.lineage = []
            
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "ModelCard":
        return cls(**data)
    

class ModelRegistry:
    """Registry for optimized models"""
    
    def __init__(self, registry_path: str = "./model_registry.json"):
        self.registry_path = Path(registry_path)
        self.models = self._load()
        
    def _load(self) -> Dict:
        if self.registry_path.exists():
            with open(self.registry_path) as f:
                return json.load(f)
        return {}
        
    def _save(self):
        with open(self.registry_path, 'w') as f:
            json.dump(self.models, f, indent=2)
            
    def register(self, card: ModelCard):
        self.models[card.model_id] = card.to_dict()
        self._save()
        logger.info(f"Registered: {card.model_id}")
        
    def get(self, model_id: str) -> Optional[ModelCard]:
        data = self.models.get(model_id)
        return ModelCard.from_dict(data) if data else None
        
    def list_models(self, target: str = None) -> list:
        if target:
            return [m for m in self.models.values() 
                   if m.get('optimization_target') == target]
        return list(self.models.values())
        
    def delete(self, model_id: str):
        if model_id in self.models:
            del self.models[model_id]
            self._save()
            
    def get_best(self, target: str, metric: str = "compression_ratio") -> Optional[ModelCard]:
        candidates = self.list_models(target)
        if not candidates:
            return None
        return max(candidates, key=lambda m: m.get('metrics', {}).get(metric, 0))


def create_model_card(
    model: Any,
    config: "OptimizationConfig",
    result: "OptimizationResult"
) -> ModelCard:
    """Create model card from optimization result"""
    return ModelCard(
        model_id=f"predycat_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        name=getattr(model, 'name', 'unknown'),
        base_model=getattr(model, 'base_model', 'unknown'),
        optimization_target=config.target_platform,
        created_at=datetime.now().isoformat(),
        metrics={
            "compression_ratio": result.compression_ratio,
            "latency_improvement": result.latency_improvement,
            "accuracy_drop": result.accuracy_drop,
            "original_size_mb": result.original_size_mb,
            "optimized_size_mb": result.optimized_size_mb,
        },
        pipeline_steps=result.pipeline_steps,
        config=asdict(config)
    )


def export_model_card(card: ModelCard, output_path: str = "model_card.json"):
    """Export model card to file"""
    with open(output_path, 'w') as f:
        json.dump(card.to_dict(), f, indent=2)
    return output_path


def import_model_card(input_path: str) -> ModelCard:
    """Import model card from file"""
    with open(input_path) as f:
        return ModelCard.from_dict(json.load(f))