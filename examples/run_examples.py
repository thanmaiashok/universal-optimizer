"""Example runs for OptiLLM Universal Optimizer"""

import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def example_1_simple_model():
    """Example 1: Optimize a simple PyTorch model"""
    logger.info("="*50)
    logger.info("Example 1: Simple Model Optimization")
    logger.info("="*50)
    
    import torch
    import torch.nn as nn
    
    class SimpleTransformer(nn.Module):
        def __init__(self, vocab_size=32000, hidden_size=512, num_layers=6):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, hidden_size)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=8,
                dim_feedforward=hidden_size * 4,
                batch_first=True,
                dropout=0.1
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
            self.output = nn.Linear(hidden_size, vocab_size)
            
        def forward(self, x):
            x = self.embedding(x)
            x = self.encoder(x)
            return self.output(x)
    
    model = SimpleTransformer()
    params = sum(p.numel() for p in model.parameters())
    logger.info(f"Original model: {params:,} parameters")
    
    from opti_llm.core.optimizer import OptiLLMOptimizer, OptimizationConfig
    
    config = OptimizationConfig(
        target_platform="mobile",
        preset="balanced",
        max_accuracy_drop=2.0
    )
    
    optimizer = OptiLLMOptimizer(config=config, output_dir="./outputs/example1")
    result = optimizer.optimize(model)
    
    if result.success:
        logger.info(f"SUCCESS!")
        logger.info(f"  Size: {result.original_size_mb:.1f}MB → {result.optimized_size_mb:.1f}MB ({result.compression_ratio:.1f}x)")
        logger.info(f"  Latency: {result.original_latency_ms:.1f}ms → {result.optimized_latency_ms:.1f}ms ({result.latency_improvement:.1f}x faster)")
    else:
        logger.error(f"FAILED: {result.error}")


def example_2_quantization():
    """Example 2: Manual quantization"""
    logger.info("="*50)
    logger.info("Example 2: Quantization")
    logger.info("="*50)
    
    import torch
    import torch.nn as nn
    
    model = nn.Sequential(
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Linear(4096, 32000)
    )
    
    from opti_llm.core.quantizer import QuantizationEngine, QuantType
    
    engine = QuantizationEngine()
    
    for qtype in [QuantType.FP16, QuantType.INT8, QuantType.INT4]:
        try:
            quantized_model, result = engine.quantize(model, quant_type=qtype)
            logger.info(f"  {qtype.value}: {result.original_size_mb:.1f}MB → {result.quantized_size_mb:.1f}MB ({result.compression_ratio:.1f}x)")
            del quantized_model
        except Exception as e:
            logger.warning(f"  {qtype.value}: Failed - {e}")


def example_3_profiling():
    """Example 3: Model profiling"""
    logger.info("="*50)
    logger.info("Example 3: Profiling")
    logger.info("="*50)
    
    import torch
    import torch.nn as nn
    
    model = nn.Sequential(
        nn.Linear(512, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.Linear(512, 10)
    )
    
    from opti_llm.core.profiler import MultiProfiler, Platform
    
    profiler = MultiProfiler()
    
    results = profiler.profile_full(model, input_shape=(1, 128))
    
    if results.laptop:
        logger.info(f"  Laptop: {results.laptop.latency_ms:.2f}ms, VRAM: {results.laptop.vram_mb:.1f}MB")
    if results.mobile:
        logger.info(f"  Mobile: {results.mobile.latency_ms:.2f}ms, RAM: {results.mobile.ram_mb:.1f}MB, Thermal: {results.mobile.thermal_throttle_risk:.1f}")
    if results.edge:
        logger.info(f"  Edge: {results.edge.latency_ms:.2f}ms, RAM: {results.edge.ram_mb:.1f}MB")


def example_4_strategy():
    """Example 4: Strategy selection"""
    logger.info("="*50)
    logger.info("Example 4: Strategy Selection")
    logger.info("="*50)
    
    from opti_llm.core.strategy import StrategySelector, TargetPlatform, OptimizationPreset
    
    selector = StrategySelector()
    
    platforms = [TargetPlatform.LAPTOP, TargetPlatform.MOBILE, TargetPlatform.CLOUD]
    presets = [OptimizationPreset.BALANCED, OptimizationPreset.ULTRA_COMPRESSION, OptimizationPreset.MAX_ACCURACY]
    
    for platform in platforms:
        for preset in presets:
            strategy = selector.select_strategy(platform, preset)
            logger.info(f"  {platform.value}/{preset.value}:")
            logger.info(f"    Pipeline: {[s.value for s in strategy.steps]}")
            logger.info(f"    Quantize: {strategy.quantize_type}, Prune: {strategy.prune_ratio}")


def example_5_reasoning_evaluation():
    """Example 5: Reasoning evaluation"""
    logger.info("="*50)
    logger.info("Example 5: Reasoning Evaluation")
    logger.info("="*50)
    
    from opti_llm.core.evaluator import ReasoningEvaluator, ReasoningTaskType
    
    evaluator = ReasoningEvaluator()
    
    score = evaluator.evaluate_reasoning(None, num_prompts=2)
    
    logger.info(f"  Coherence: {score.coherence:.2f}")
    logger.info(f"  Logical: {score.logical_correctness:.2f}")
    logger.info(f"  Hallucination: {score.hallucination_rate:.2f}")
    logger.info(f"  Overall: {score.overall_score:.2f}")


def example_6_hardware():
    """Example 6: Hardware info"""
    logger.info("="*50)
    logger.info("Example 6: Hardware Information")
    logger.info("="*50)
    
    from opti_llm.utils import get_hardware_info, get_memory_stats
    
    hw = get_hardware_info()
    logger.info(f"  CPU: {hw.cpu}")
    logger.info(f"  RAM: {hw.ram_gb:.1f}GB")
    logger.info(f"  CUDA: {hw.has_cuda} ({hw.cuda_version})")
    logger.info(f"  MPS: {hw.has_mps}")
    
    mem = get_memory_stats()
    if mem.get("vram_total_gb", 0) > 0:
        logger.info(f"  VRAM: {mem.get('vram_allocated_gb', 0):.1f}/{mem.get('vram_total_gb', 0):.1f}GB")
    logger.info(f"  RAM: {mem.get('ram_available_gb', 0):.1f}/{mem.get('ram_total_gb', 0):.1f}GB")


def example_7_cli():
    """Example 7: CLI usage"""
    logger.info("="*50)
    logger.info("Example 7: CLI Usage")
    logger.info("="*50)
    
    logger.info("""
Usage:
  optillm optimize <model> --target mobile --preset balanced
  optillm optimize <model> --target mobile --max-drop 2%
  optillm optimize <model> --target mobile --formats onnx,pt
  optillm profile <model>
  optillm evaluate <model>

Examples:
  optillm optimize gpt2 --target mobile
  optillm optimize ./model.pt --target laptop --preset ultra_compression
  optillm batch "models/*.pt" --target mobile
""")


def run_all_examples():
    examples = [
        example_1_simple_model,
        example_2_quantization,
        example_3_profiling,
        example_4_strategy,
        example_5_reasoning_evaluation,
        example_6_hardware,
        example_7_cli,
    ]
    
    for i, example in enumerate(examples, 1):
        try:
            example()
        except Exception as e:
            logger.error(f"Example {i} failed: {e}")
        print()


if __name__ == "__main__":
    logger.info("Running OptiLLM Examples")
    run_all_examples()